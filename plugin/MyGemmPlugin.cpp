#include "MyGemmPlugin.h"
#include <cstring>
#include <iostream>
#include <cuda_runtime.h> // 需要 cudaMalloc

using namespace nvinfer1;
using namespace nvinfer1::myplugin;

// 更新 Launcher 声明，必须有 9 个参数！
extern "C" void matrixMulPhase6_AsyncLauncher(
    float* A, float* B, float* C, 
    const float* Bias, int use_relu, 
    int M, int N, int K, cudaStream_t stream
);

// Helper
template <typename T> void write(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val; buffer += sizeof(T);
}
template <typename T> void read(const char*& buffer, T& val) {
    val = *reinterpret_cast<const T*>(buffer); buffer += sizeof(T);
}

// ==========================================
// 1. 构造函数 (Host 端的初始化)
// ==========================================
MyGemmPlugin::MyGemmPlugin(const std::string& name, int N, int K, const std::vector<float>& bias)
    : mLayerName(name), mN(N), mK(K), mHostBias(bias)
{
    // 如果传入的 Bias 为空，我们初始化全0，防止崩溃
    if (mHostBias.empty()) {
        mHostBias.resize(N, 0.0f);
    }
}

// 反序列化构造函数 (从 Engine 文件恢复)
MyGemmPlugin::MyGemmPlugin(const std::string& name, const void* data, size_t length)
    : mLayerName(name)
{
    const char* d = reinterpret_cast<const char*>(data);
    read(d, mN);
    read(d, mK);
    
    // 恢复 Bias 数据
    mHostBias.resize(mN);
    // 从 buffer 里直接拷贝数据到 vector
    std::memcpy(mHostBias.data(), d, mN * sizeof(float));
    d += mN * sizeof(float);
}

// ==========================================
// 2. 资源管理 (Device 端的初始化)
// ==========================================

// initialize: 引擎启动时调用 (比如 context 创建时)
int MyGemmPlugin::initialize() noexcept
{
    if (mN > 0) {
        // 防止重复申请内存导致内存泄漏
        if (mDeviceBias != nullptr) return 0;

        cudaMalloc(&mDeviceBias, mN * sizeof(float));
        cudaMemcpy(mDeviceBias, mHostBias.data(), mN * sizeof(float), cudaMemcpyHostToDevice);

        // [Debug] 打印当前对象地址
        std::cout << "DEBUG: [" << this << "] initialize() done. Bias Size: " << mN << std::endl;
    }
    return 0;
}

// terminate: 引擎销毁时调用
void MyGemmPlugin::terminate() noexcept 
{
    if (mDeviceBias) {
        cudaFree(mDeviceBias);
        mDeviceBias = nullptr;
    }
}

// ==========================================
// 3. 执行逻辑
// ==========================================
int MyGemmPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    int M = inputDesc[0].dims.d[0];
    const float* d_A = reinterpret_cast<const float*>(inputs[0]);
    const float* d_B = reinterpret_cast<const float*>(inputs[1]);
    float* d_C = reinterpret_cast<float*>(outputs[0]);

    // ---------------------------------------------------------
    // [关键修复] 惰性初始化检查 (JIT Check)
    // ---------------------------------------------------------
    if (mDeviceBias == nullptr) {
        std::cout << "WARNING: [" << this << "] mDeviceBias is NULL in enqueue! Triggering Lazy Init..." << std::endl;

        // 尝试亡羊补牢
        // 注意：这里必须用 const_cast 因为 enqueue 是 const 函数，但我们需要修改成员变量
        // 这是一个 Dirty Hack，但在 Debug 阶段非常有效
        const_cast<MyGemmPlugin*>(this)->initialize();

        if (mDeviceBias == nullptr) {
            std::cout << "ERROR: Lazy Init failed! Bias is still NULL." << std::endl;
        }
    }
    // ---------------------------------------------------------

    // [Debug] 打印发射信息
    // std::cout << "[CPU Enqueue] Obj: " << this << " Bias Ptr: " << mDeviceBias << std::endl;

    matrixMulPhase6_AsyncLauncher(
        const_cast<float*>(d_A),
        const_cast<float*>(d_B),
        d_C,
        mDeviceBias,
        1, // use_relu = true
        M, mN, mK,
        stream
    );
    return 0;
}

// ==========================================
// 4. 序列化 (存档)
// ==========================================
size_t MyGemmPlugin::getSerializationSize() const noexcept {
    // N + K + Bias数据本身
    return sizeof(mN) + sizeof(mK) + (mN * sizeof(float));
}

void MyGemmPlugin::serialize(void* buffer) const noexcept {
    char* d = reinterpret_cast<char*>(buffer);
    write(d, mN);
    write(d, mK);
    // 写入 Bias 数组内容
    std::memcpy(d, mHostBias.data(), mN * sizeof(float));
    d += mN * sizeof(float);
}

// --- 标准样板代码 ---
int MyGemmPlugin::getNbOutputs() const noexcept { return 1; }
DimsExprs MyGemmPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
    DimsExprs output; output.nbDims = 2;
    output.d[0] = inputs[0].d[0]; output.d[1] = exprBuilder.constant(mN);
    return output;
}
size_t MyGemmPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept { return 0; }
bool MyGemmPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
}
const char* MyGemmPlugin::getPluginType() const noexcept { return "MyGemmPlugin"; }
const char* MyGemmPlugin::getPluginVersion() const noexcept { return "1"; }
void MyGemmPlugin::destroy() noexcept { delete this; }
IPluginV2DynamicExt* MyGemmPlugin::clone() const noexcept {
    auto* p = new MyGemmPlugin(mLayerName, mN, mK, mHostBias);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}
void MyGemmPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* MyGemmPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
DataType MyGemmPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept { return DataType::kFLOAT; }
void MyGemmPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbIn, const DynamicPluginTensorDesc* out, int nbOut) noexcept {}


// ==========================================
// 5. Creator (支持从 Python 传入 Bias)
// ==========================================
PluginFieldCollection MyGemmPluginCreator::mFC{};
std::vector<PluginField> MyGemmPluginCreator::mPluginAttributes;

MyGemmPluginCreator::MyGemmPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("N", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    // 增加 Bias 属性，类型是 Float32
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1)); 
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MyGemmPluginCreator::getPluginName() const noexcept { return "MyGemmPlugin"; }
const char* MyGemmPluginCreator::getPluginVersion() const noexcept { return "1"; }
const PluginFieldCollection* MyGemmPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* MyGemmPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    int N = 0, K = 0;
    std::vector<float> bias;

    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "N")) N = *(static_cast<const int*>(fields[i].data));
        else if (!strcmp(attrName, "K")) K = *(static_cast<const int*>(fields[i].data));
        else if (!strcmp(attrName, "bias")) {
            // 获取数组长度和数据指针
            int len = fields[i].length;
            const float* ptr = static_cast<const float*>(fields[i].data);
            bias.assign(ptr, ptr + len);
            std::cout << "DEBUG: createPlugin found bias! Length: " << len << std::endl;
        }
    }
    return new MyGemmPlugin(name, N, K, bias);
}

IPluginV2* MyGemmPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    return new MyGemmPlugin(name, serialData, serialLength);
}

void MyGemmPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* MyGemmPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(MyGemmPluginCreator);