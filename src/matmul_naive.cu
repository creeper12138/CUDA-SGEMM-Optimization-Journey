#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_pipeline.h>


// --- Phase 4 核心参数 ---
// 只要动这几个参数，整个逻辑自动调整
#define BM 128 // Block M: 一个 Block 负责输出 128 行
#define BN 128 // Block N: 一个 Block 负责输出 128 列
#define BK 8   // Block K: 每次切片搬运的深度 (这也是 Shared Mem 的 K 维)
#define TM 8   // Thread M: 一个线程负责算 8 行
#define TN 8   // Thread N: 一个线程负责算 8 列

#define BLOCK_SIZE 32


// 错误检查宏
#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


__global__ void matrixMulPhase4G(float* A, float* B, float* C, int M, int N, int K) {
    // [1] 索引计算
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * blockDim.x + tx; // 0 ~ 511

    // [2] 寄存器分配
    // 8x4 = 32 个元素，寄存器压力适中 (~50-60 range)
    float thread_results[TM][TN] = { 0.0f };
    float reg_a[TM];
    float reg_b[TN];

    // [3] 双缓冲 Shared Memory [2]
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    // [4] 流水线搬运变量
    float4 load_a_reg;
    float4 load_b_reg;
    int write_stage = 1;
    int read_stage = 0;

    // 指针定位
    float* A_ptr = A + by * BM * K;
    float* B_ptr = B + bx * BN;

    // ==============================================================
    // Prologue: 预取第 0 个块 (写入 write_stage = 0 号盘子)
    // ==============================================================
    // 512 个线程中，前 256 个负责搬运
    if (tid < 256) {
        // --- A: 128x8 ---
        int load_a_row = tid / 2;
        int load_a_col = (tid % 2) * 4;

        float* src_ptr_a = &A_ptr[load_a_row * K + 0 + load_a_col];
        float4 v_a = reinterpret_cast<float4*>(src_ptr_a)[0];

        As[0][load_a_row][load_a_col + 0] = v_a.x;
        As[0][load_a_row][load_a_col + 1] = v_a.y;
        As[0][load_a_row][load_a_col + 2] = v_a.z;
        As[0][load_a_row][load_a_col + 3] = v_a.w;

        // --- B: 8x128 ---
        int load_b_row = tid / 32;
        int load_b_col = (tid % 32) * 4;

        float* src_ptr_b = &B_ptr[(0 + load_b_row) * N + load_b_col];
        float4 v_b = reinterpret_cast<float4*>(src_ptr_b)[0];

        Bs[0][load_b_row][load_b_col + 0] = v_b.x;
        Bs[0][load_b_row][load_b_col + 1] = v_b.y;
        Bs[0][load_b_row][load_b_col + 2] = v_b.z;
        Bs[0][load_b_row][load_b_col + 3] = v_b.w;
    }

    __syncthreads(); // 必须同步：确保第0块数据就位

    // ==============================================================
    // Main Loop: 流水线
    // ==============================================================
    for (int k = 0; k < K; k += BK) {

        // ----------------------------------------------------------
        // [Step 1] 预取下一块 (Global -> Register)
        // ----------------------------------------------------------
        int next_k = k + BK;
        if (tid < 256 && next_k < K) {
            // Load A
            int load_a_row = tid / 2;
            int load_a_col = (tid % 2) * 4;
            float* src_ptr_a = &A_ptr[load_a_row * K + next_k + load_a_col];
            load_a_reg = reinterpret_cast<float4*>(src_ptr_a)[0];

            // Load B
            int load_b_row = tid / 32;
            int load_b_col = (tid % 32) * 4;
            float* src_ptr_b = &B_ptr[(next_k + load_b_row) * N + load_b_col];
            load_b_reg = reinterpret_cast<float4*>(src_ptr_b)[0];
        }

        // ----------------------------------------------------------
        // [Step 2] 计算当前块 (Shared[read_stage] -> Accum)
        // ----------------------------------------------------------
#pragma unroll
        for (int i = 0; i < BK; ++i) {
            // 加载 A 的一小列 (8个数)
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                reg_a[r] = As[read_stage][ty * TM + r][i];
            }
            // 加载 B 的一小行 (4个数)
#pragma unroll
            for (int c = 0; c < TN; ++c) {
                reg_b[c] = Bs[read_stage][i][tx * TN + c];
            }
            // 外积计算 (8x4=32 FMA)
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                for (int c = 0; c < TN; ++c) {
                    thread_results[r][c] += reg_a[r] * reg_b[c];
                }
            }
        }

        // ----------------------------------------------------------
        // [Step 3] 填入下一块 (Register -> Shared[write_stage])
        // ----------------------------------------------------------

        // 同步 1：确保大家算完了 read_stage
        __syncthreads();

        if (tid < 256 && next_k < K) {
            // A -> As[write_stage]
            int load_a_row = tid / 2;
            int load_a_col = (tid % 2) * 4;
            As[write_stage][load_a_row][load_a_col + 0] = load_a_reg.x;
            As[write_stage][load_a_row][load_a_col + 1] = load_a_reg.y;
            As[write_stage][load_a_row][load_a_col + 2] = load_a_reg.z;
            As[write_stage][load_a_row][load_a_col + 3] = load_a_reg.w;

            // B -> Bs[write_stage]
            int load_b_row = tid / 32;
            int load_b_col = (tid % 32) * 4;
            Bs[write_stage][load_b_row][load_b_col + 0] = load_b_reg.x;
            Bs[write_stage][load_b_row][load_b_col + 1] = load_b_reg.y;
            Bs[write_stage][load_b_row][load_b_col + 2] = load_b_reg.z;
            Bs[write_stage][load_b_row][load_b_col + 3] = load_b_reg.w;
        }

        // 乒乓切换 (if inside loop to avoid divergence outside, safe here)
        if (next_k < K) {
            read_stage ^= 1;
            write_stage ^= 1;
        }

        // 同步 2：确保大家写完了 write_stage
        __syncthreads();
    }

    // [5] 写回 Global Memory
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        for (int c = 0; c < TN; ++c) {
            int global_row = by * BM + ty * TM + r;
            int global_col = bx * BN + tx * TN + c;

            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = thread_results[r][c];
            }
        }
    }
}

__global__ void matrixMulPhase4(float* A, float* B, float* C, int M, int N, int K) {
    // [1] 既然 Block 是 128x128，而线程每人算 8x8
    // 线程块的维度 (blockDim) 必须是 (128/TN, 128/TM) = (16, 16)
    // 总线程数 = 256

    // 算出该 Block 在全局 C 矩阵中的左上角坐标
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // 线程索引
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * blockDim.x + tx; // 0 ~ 255 的线性 ID

    // [2] 寄存器分配 (私有财产)
    // 累加器：从头拿到尾，算完才写回显存
    float thread_results[TM][TN] = { 0.0f };
    // 缓存器：每次用来做外积计算
    float reg_a[TM];
    float reg_b[TN];

    // [3] Shared Memory 分配 (公共财产)
    // As: [128行][8列], Bs: [8行][128列]
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // [4] 指针定位
    // A_ptr 指向当前 Block 负责的 A 行的起始位置
    // B_ptr 指向当前 Block 负责的 B 列的起始位置
    float* A_ptr = A + by * BM * K;
    float* B_ptr = B + bx * BN;

    // --- 最外层循环 (Global K Loop): 每次前进 BK(8) 步 ---
    for (int k = 0; k < K; k += BK) {

        // === 阶段 1: 搬运 (Global -> Shared) ===
        // 目标：256 个线程合力搬运 128x8 = 1024 个数
        // 策略：每人搬 4 个数 (模拟 float4)

        // --- 搬运 A (128行 x 8列) ---
        // 把 256 个线程映射到 128 行，每行只需要 2 个线程 (2x4=8)
        // load_a_row: 0~127
        // load_a_col: 0 或 4
        int load_a_row = tid / 2;
        int load_a_col = (tid % 2) * 4;

        // 边界检查 (防止最后一次 K 不足 8 时越界，虽然通常还是会补齐 padding)
        if (load_a_row < BM) {
#pragma unroll
                float* src_ptr = &A_ptr[load_a_row * K + k + load_a_col];

                // 向量化读取：一次读 128 bit
                float4 v_a = reinterpret_cast<float4*>(src_ptr)[0];

                // 写入 Shared Memory (拆包)
                // 也就是把 v_a.x, v_a.y, v_a.z, v_a.w 填进去
                As[load_a_row][load_a_col + 0] = v_a.x;
                As[load_a_row][load_a_col + 1] = v_a.y;
                As[load_a_row][load_a_col + 2] = v_a.z;
                As[load_a_row][load_a_col + 3] = v_a.w;
            
        }

        // --- 搬运 B (8行 x 128列) ---
        // 把 256 个线程映射到 8 行，每行需要 32 个线程 (32x4=128)
        // load_b_row: 0~7
        // load_b_col: 0, 4, 8 ... 124
        int load_b_row = tid / 32;
        int load_b_col = (tid % 32) * 4;

        if (load_b_row < BK) {
#pragma unroll
            float* src_ptr = &B_ptr[(k + load_b_row) * N + load_b_col];

            // 向量化读取
            float4 v_b = reinterpret_cast<float4*>(src_ptr)[0];

            // 写入 Shared Memory
            Bs[load_b_row][load_b_col + 0] = v_b.x;
            Bs[load_b_row][load_b_col + 1] = v_b.y;
            Bs[load_b_row][load_b_col + 2] = v_b.z;
            Bs[load_b_row][load_b_col + 3] = v_b.w;
        }

        __syncthreads(); // 等待货物到齐

        // === 阶段 2: 计算 (Shared -> Register 外积) ===
        // 中间层循环: 遍历切片深度 (0~7)
#pragma unroll
        for (int i = 0; i < BK; ++i) {

            // a. 加载 A 的一小列 (TM=8) 到寄存器
            // 既然 thread_results[r][c] 对应的是 As[ty*TM + r]
            // 我们要拿的就是 As 的第 i 列，行号是 "我负责的那8行"
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                reg_a[r] = As[ty * TM + r][i];
            }

            // b. 加载 B 的一小行 (TN=8) 到寄存器
            // 同理，拿 Bs 的第 i 行，列号是 "我负责的那8列"
#pragma unroll
            for (int c = 0; c < TN; ++c) {
                reg_b[c] = Bs[i][tx * TN + c];
            }

            // c. 外积计算 (64 次 FMA)
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                for (int c = 0; c < TN; ++c) {
                    thread_results[r][c] += reg_a[r] * reg_b[c];
                }
            }
        }
        __syncthreads(); // 等待大家算完，才能覆盖 Shared Memory 进行下一轮搬运
    }

    // [5] 写回 (Register -> Global)
    // 每个线程把自己的 8x8 = 64 个结果写回 C
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        for (int c = 0; c < TN; ++c) {
            int global_row = by * BM + ty * TM + r;
            int global_col = bx * BN + tx * TN + c;

            // 只有在范围内的才写回 (防止 N 不是 128 倍数的情况)
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = thread_results[r][c];
            }
        }
    }
}

__global__ void matrixMulPhase6_Async(float* A, float* B, float* C, int M, int N, int K) {
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;

    // [1] 寄存器分配
    // 回到 8x8 = 64 个累加器。
    // 但是！我们不需要 load_a_reg/load_b_reg 了！这就省了 8~16 个寄存器。
    float thread_results[TM][TN] = { 0.0f };
    float reg_a[TM];
    float reg_b[TN];

    // [2] Shared Memory (保留 padding 习惯，防止 Bank Conflict)
    __shared__ float As[2][BM][BK]; // 也可以加 padding [BK+1]
    __shared__ float Bs[2][BK][BN];

    // [3] 指针定位
    float* A_ptr = A + by * BM * K;
    float* B_ptr = B + bx * BN;

    // [4] 流水线控制
    // 不需要 load_reg 变量了！
    int write_stage = 1;
    int read_stage = 0;

    // ==============================================================
    // Prologue: 预取第 0 个块
    // ==============================================================

    // 我们用 256 个线程搬运。A需要 128x8=1024 float。
    // 每个 float4 是 4 个 float。需要 256 个线程。刚好每人搬一个 float4。
    // cp.async 只需要指定：目标Shared指针，源Global指针，大小(16字节=float4)

    // 发起异步搬运指令 (A)
    {
        int load_a_row = tid / 2;
        int load_a_col = (tid % 2) * 4;
        void* dest = &As[0][load_a_row][load_a_col];
        void* src = &A_ptr[load_a_row * K + 0 + load_a_col];
        // 16 表示 16 bytes = sizeof(float4)
        __pipeline_memcpy_async(dest, src, 16);
    }

    // 发起异步搬运指令 (B)
    {
        int load_b_row = tid / 32;
        int load_b_col = (tid % 32) * 4;
        void* dest = &Bs[0][load_b_row][load_b_col];
        void* src = &B_ptr[(0 + load_b_row) * N + load_b_col];
        __pipeline_memcpy_async(dest, src, 16);
    }

    // 提交这一批次的搬运任务
    __pipeline_commit();

    // 等待这一批次完成 (wait_prior(0) 表示等待剩下 0 个没完成，即全完成)
    __pipeline_wait_prior(0);

    __syncthreads(); // 必须同步，防止有人计算跑太快

    // ==============================================================
    // Main Loop
    // ==============================================================
    for (int k = 0; k < K; k += BK) {

        // ----------------------------------------------------------
        // [Step 1] 预取下一块 (Direct Global -> Shared)
        // ----------------------------------------------------------
        int next_k = k + BK;
        if (next_k < K) {
            // A -> As[write_stage]
            int load_a_row = tid / 2;
            int load_a_col = (tid % 2) * 4;
            void* dest = &As[write_stage][load_a_row][load_a_col];
            void* src = &A_ptr[load_a_row * K + next_k + load_a_col];
            __pipeline_memcpy_async(dest, src, 16);

            // B -> Bs[write_stage]
            int load_b_row = tid / 32;
            int load_b_col = (tid % 32) * 4;
            dest = &Bs[write_stage][load_b_row][load_b_col];
            src = &B_ptr[(next_k + load_b_row) * N + load_b_col];
            __pipeline_memcpy_async(dest, src, 16);
        }

        // 提交搬运任务：告诉硬件“这一轮的搬运指令发完了，打包成一个组”
        __pipeline_commit();

        // ----------------------------------------------------------
        // [Step 2] 计算当前块
        // ----------------------------------------------------------
        // 这时候，硬件正在后台疯狂搬运 next_k 的数据
        // 我们利用这段时间做 8x8 的高强度计算

#pragma unroll
        for (int i = 0; i < BK; ++i) {
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                reg_a[r] = As[read_stage][ty * TM + r][i];
            }
#pragma unroll
            for (int c = 0; c < TN; ++c) {
                reg_b[c] = Bs[read_stage][i][tx * TN + c];
            }
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                for (int c = 0; c < TN; ++c) {
                    thread_results[r][c] += reg_a[r] * reg_b[c];
                }
            }
        }

        // ----------------------------------------------------------
        // [Step 3] 流水线等待
        // ----------------------------------------------------------

        // 这一步非常关键！
        // wait_prior(0) 表示：等待刚才提交的那一组 (next_k) 彻底完成
        // 因为我们马上循环回来就要用它了
        // *实际上，更高级的优化是 wait_prior(1)，做多级流水，但双缓冲 wait_prior(0) 足够了
        __pipeline_wait_prior(0);

        __syncthreads(); // 确保所有线程都读完了 read_stage，也确保 write_stage 已经填满了

        // 交换
        if (next_k < K) {
            read_stage ^= 1;
            write_stage ^= 1;
        }
    }

    // [5] 写回
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        for (int c = 0; c < TN; ++c) {
            int global_row = by * BM + ty * TM + r;
            int global_col = bx * BN + tx * TN + c;

            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = thread_results[r][c];
            }
        }
    }
}


extern "C" void matrixMulPhase6_AsyncLauncher(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + 127) / 128, (M + 127) / 128); // 注意这里的 Grid 计算

    // 这里的 stream 是从 TRT 传进来的，必须用它！
    matrixMulPhase6_Async << <grid, block, 0, stream >> > (A, B, C, M, N, K);
}




// CPU 参考实现 (验证用，非常慢，只用来算小规模或者验证)
void matrixMulCPU(const float* A, const float* B, float* C, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// 验证函数
bool verifyResult(const float* cpu_res, const float* gpu_res, int size) {
    for (int i = 0; i < size; i++) {
        // 矩阵乘法累加误差会大一些，容忍度调高到 1e-3
        if (std::abs(cpu_res[i] - gpu_res[i]) > 1e-3) {
            printf("FAILED at index %d: CPU=%f, GPU=%f\n", i, cpu_res[i], gpu_res[i]);
            return false;
        }
    }
    return true;
}

// 在 main 函数之前添加
void printPerformanceMetrics(float avg_ms, int M, int N, int K, float devicePeakGFLOPS, float devicePeakBW) {
    // 1. 计算实测 GFLOPS
    double totalOps = 2.0 * (double)M * (double)N * (double)K;
    double measuredGFLOPS = (totalOps * 1e-9) / (avg_ms / 1000.0f);

    // 2. 计算实测有效带宽 (Effective Bandwidth)
    // 假设 A, B 读一次，C 写一次 (理想情况)
    double totalBytes = ((double)M * K + (double)K * N + (double)M * N) * sizeof(float);
    double measuredBW = (totalBytes * 1e-9) / (avg_ms / 1000.0f);

    // 3. 计算算术强度 (Arithmetic Intensity)
    double intensity = totalOps / totalBytes;

    printf("\n===== Performance Report =====\n");
    printf("Matrix Size         : %d x %d x %d\n", M, N, K);
    printf("Time                : %.3f ms\n", avg_ms);
    printf("Achieved Compute    : %.2f GFLOPS\n", measuredGFLOPS);
    printf("Achieved Bandwidth  : %.2f GB/s (Effective)\n", measuredBW);
    printf("Arithmetic Intensity: %.2f FLOPs/Byte\n", intensity);

    // 4. 计算利用率 (如果提供了硬件参数)
    if (devicePeakGFLOPS > 0) {
        printf("Compute Utilization : %.2f %% (Target: >70%%)\n", (measuredGFLOPS / devicePeakGFLOPS) * 100);
    }
    if (devicePeakBW > 0) {
        // 对于 MatMul，这个数值不需要很高，因为它不是瓶颈
        printf("Memory Utilization  : %.2f %% (Reference)\n", (measuredBW / devicePeakBW) * 100);
    }
    printf("==============================\n");
}

int main() {
    // 1. 设置矩阵大小
    // 1024 x 1024 是一个分水岭。
    // 计算量 = 1024^3 * 2 (乘加) = 2 GFLOPs
    int N = 4096;
    size_t bytes = N * N * sizeof(float);

    printf("Matrix Size: %d x %d\n", N, N);

    // 2. Host 内存准备
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_cpu(N * N);
    std::vector<float> h_C_gpu(N * N);

    // 初始化
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 3. Device 内存准备
    float* d_A, * d_B, * d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // 4. 配置 Grid/Block (使用 2D)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + BM-1) / BM
        ,
        (N + BN-1) / BN
    );

    // 5. 运行 GPU Kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running Naive GPU implementation...\n");
    cudaEventRecord(start);

    // 跑 10 次取平均，减少波动
    int loops = 1;
    for (int i = 0; i < loops; i++) {
        matrixMulPhase6_Async << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N, N, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / loops;

    printf("Avg GPU Time: %.3f ms\n", avg_ms);
    // 理论 FP32 算力 = 2 * 3072 * 2370 MHz ≈ 14,561 GFLOPS
    float myCardGFLOPS = 14500.0f;

    // 理论带宽 = 128 bit * 16 Gbps / 8 = 256 GB/s
    float myCardBW = 256.0f;

    printPerformanceMetrics(avg_ms, N, N, N, myCardGFLOPS, myCardBW);    // 计算 GFLOPS (Giga Floating-point Operations Per Second)
    // 矩阵乘法总浮点运算次数 = 2 * N^3
    double ops = 2.0 * pow(N, 3);
    double gflops = (ops * 1e-9) / (avg_ms / 1000.0f);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // 6. 验证结果 (只在 N 较小时验证，或者只验证这一轮)
    // 注意：CPU 跑 1024x1024 可能会花几秒钟
    // --- 在 main 函数末尾添加 ---

    printf("\nRunning Random Spot Check verification...\n");

    // 1. 把 GPU 结果拷回 CPU (只拷回我们需要验证的那几个点也行，但全拷回来写代码简单点)
    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    int check_count = 10; // 随机抽查 10 个点

    for (int i = 0; i < check_count; ++i) {
        // 随机生成坐标
        int row = rand() % N;
        int col = rand() % N;

        // 2. CPU 手动算这一个点 (只算这一行乘这一列)
        float cpu_val = 0.0f;
        for (int k = 0; k < N; ++k) {
            cpu_val += h_A[row * N + k] * h_B[k * N + col];
        }

        // 3. 取出 GPU 算的结果
        float gpu_val = h_C_gpu[row * N + col];

        // 4. 比对 (允许一点点浮点误差)
        if (std::abs(cpu_val - gpu_val) > 1e-2) {
            printf("FAIL: at (%d, %d) CPU=%.4f, GPU=%.4f\n", row, col, cpu_val, gpu_val);
            errors++;
        }
        else {
            printf("PASS: at (%d, %d) matched within error.\n", row, col);
        }
    }

    if (errors == 0) printf("verification successful! (Checked %d random points)\n", check_count);


    if (N <= 1024) {
        printf("Running CPU implementation for verification (might take time)...\n");
        matrixMulCPU(h_A.data(), h_B.data(), h_C_cpu.data(), N);


        CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        if (verifyResult(h_C_cpu.data(), h_C_gpu.data(), N * N)) {
            printf("Result: PASSED\n");
        }

    }
    else {
        printf("Skipping CPU verification for large N (%d)\n", N);
    }
    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}