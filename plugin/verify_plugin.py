import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# 加载 DLL
PLUGIN_LIB_PATH = "./MyGemmPlugin.dll"
ctypes.CDLL(PLUGIN_LIB_PATH)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# ================================================================
# 构建引擎：对比 "你的融合插件" vs "官方原生组合拳"
# ================================================================
def build_engine(M, N, K, bias_data, use_custom_plugin=True):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # 1. 定义输入 A, B
    input_a = network.add_input(name="InputA", dtype=trt.float32, shape=(M, K))
    input_b = network.add_input(name="InputB", dtype=trt.float32, shape=(K, N))

    if use_custom_plugin:
        # --- 方案 A: 你的融合插件 (Fused Plugin) ---
        # 逻辑：一次 Kernel Launch 完成 GEMM + Bias + ReLU
        
        registry = trt.get_plugin_registry()
        creator = registry.get_plugin_creator("MyGemmPlugin", "1", "")
        
        # 传入 N, K
        p_n = trt.PluginField("N", np.array([N], dtype=np.int32), trt.PluginFieldType.INT32)
        p_k = trt.PluginField("K", np.array([K], dtype=np.int32), trt.PluginFieldType.INT32)
        
        # 传入 Bias 权重 (注意：这是通过 Creator 传入的，会被序列化到 Engine 里)
        # 必须把 float32 的 numpy 数组传进去
        p_bias = trt.PluginField("bias", bias_data, trt.PluginFieldType.FLOAT32)
        
        fc = trt.PluginFieldCollection([p_n, p_k, p_bias])
        plugin = creator.create_plugin("MyGemm_Fused", fc)
        
        layer = network.add_plugin_v2(inputs=[input_a, input_b], plugin=plugin)
        layer.get_output(0).name = "OutputFused"
        network.mark_output(layer.get_output(0))
        
    else:
        # --- 方案 B: 官方原生 (Native TRT) ---
        # 逻辑：MatMul -> ElementWise(Add) -> Activation(ReLU)
        
        # 1. MatMul
        mm_layer = network.add_matrix_multiply(input_a, trt.MatrixOperation.NONE, 
                                               input_b, trt.MatrixOperation.NONE)
        
        # 2. Add Bias
        # 需要把 bias_data 变成一个 Constant Layer，然后加到 MatMul 结果上
        # reshape bias to (1, N) for broadcasting
        bias_const = network.add_constant((1, N), trt.Weights(bias_data))
        add_layer = network.add_elementwise(mm_layer.get_output(0), 
                                             bias_const.get_output(0), 
                                             trt.ElementWiseOperation.SUM)
        
        # 3. ReLU
        relu_layer = network.add_activation(add_layer.get_output(0), trt.ActivationType.RELU)
        
        relu_layer.get_output(0).name = "OutputNative"
        network.mark_output(relu_layer.get_output(0))

    # 构建 Engine
    try:
        engine_bytes = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(engine_bytes)
    except AttributeError:
        return builder.build_engine(network, config)

# ================================================================
# 跑分函数
# ================================================================
def benchmark(engine, d_a, d_b, d_c, name="Task"):
    context = engine.create_execution_context()
    stream = cuda.Stream()
    bindings = [int(d_a), int(d_b), int(d_c)]

    # Warmup
    for _ in range(20):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()

    # Timing
    num_iters = 200
    start_event = cuda.Event()
    end_event = cuda.Event()

    start_event.record(stream)
    for _ in range(num_iters):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    end_event.record(stream)
    end_event.synchronize()
    
    return start_event.time_till(end_event) / num_iters

# ================================================================
# 主程序
# ================================================================
def main():
    # 设置测试规模 (大一点才能看出融合优势)
    M, N, K = 1024, 1024, 1024
    print(f"Testing Fusion (Bias+ReLU) on size {M}x{N}x{K} ...")
    
    # 准备数据
    h_a = np.random.rand(M, K).astype(np.float32)
    h_b = np.random.rand(K, N).astype(np.float32)
    # 随机生成 Bias
    h_bias = np.random.rand(N).astype(np.float32)
    
    # 申请显存
    d_a = cuda.mem_alloc(h_a.nbytes)
    d_b = cuda.mem_alloc(h_b.nbytes)
    d_c = cuda.mem_alloc(M * N * 4) # Output
    
    cuda.memcpy_htod(d_a, h_a)
    cuda.memcpy_htod(d_b, h_b)
    
    # ---------------------------------------------------------
    # 1. 验证正确性 (Correctness)
    # ---------------------------------------------------------
    print("\n>>> Checking Correctness...")
    engine_plugin = build_engine(M, N, K, h_bias, use_custom_plugin=True)
    
    # 运行插件
    context = engine_plugin.create_execution_context()
    context.execute_v2([int(d_a), int(d_b), int(d_c)])
    h_c_plugin = np.empty((M, N), dtype=np.float32)
    cuda.memcpy_dtoh(h_c_plugin, d_c)
    
    # CPU 模拟真值: MatMul + Bias + ReLU
    ref_c = np.matmul(h_a, h_b)
    ref_c = ref_c + h_bias # Broadcasting add
    ref_c = np.maximum(ref_c, 0) # ReLU
    
    diff = np.max(np.abs(h_c_plugin - ref_c))
    print(f"Max Difference: {diff}")
    
    if diff > 1e-1: # 矩阵乘法累积误差较大，容忍度设宽一点
        print("FAIL: Precision mismatch!")
        return
    else:
        print("PASS: Correctness verified!")

    # ---------------------------------------------------------
    # 2. 性能对比 (Performance)
    # ---------------------------------------------------------
    print("\n>>> Benchmarking...")
    
    # Plugin
    t_plugin = benchmark(engine_plugin, d_a, d_b, d_c, "Plugin")
    print(f"[Fused Plugin] Time: {t_plugin:.4f} ms")
    
    # Native
    engine_native = build_engine(M, N, K, h_bias, use_custom_plugin=False)
    t_native = benchmark(engine_native, d_a, d_b, d_c, "Native")
    print(f"[Native TRT ] Time: {t_native:.4f} ms")
    
    speedup = t_native / t_plugin
    print(f"\nResult: Plugin is {speedup:.2f}x speed of Native")
    
    if t_plugin < t_native:
        print("WINNER: Your Fused Plugin!")
    else:
        print("WINNER: Native TRT (Still tough to beat, but check if gap narrowed)")

if __name__ == "__main__":
    main()