#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>

namespace nvinfer1
{
    namespace myplugin
    {
        class MyGemmPlugin : public IPluginV2DynamicExt
        {
        public:
            // 修改构造函数：增加 bias 数据传入
            MyGemmPlugin(const std::string& name, int N, int K, const std::vector<float>& bias);
            MyGemmPlugin(const std::string& name, const void* data, size_t length);
            MyGemmPlugin() = delete;

            // --- 生命周期管理 (关键！) ---
            int initialize() noexcept override; // 在这里申请显存
            void terminate() noexcept override; // 在这里释放显存

            // --- 其他标准接口 ---
            int getNbOutputs() const noexcept override;
            DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
            size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
            int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
            size_t getSerializationSize() const noexcept override;
            void serialize(void* buffer) const noexcept override;
            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
            const char* getPluginType() const noexcept override;
            const char* getPluginVersion() const noexcept override;
            void destroy() noexcept override;
            IPluginV2DynamicExt* clone() const noexcept override;
            void setPluginNamespace(const char* pluginNamespace) noexcept override;
            const char* getPluginNamespace() const noexcept override;
            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
            void configurePlugin(const DynamicPluginTensorDesc* in, int nbIn, const DynamicPluginTensorDesc* out, int nbOut) noexcept override;

        private:
            std::string mLayerName;
            std::string mNamespace;
            int mN;
            int mK;

            // [新增] 权重管理
            std::vector<float> mHostBias; // CPU 端副本 (用于序列化保存)
            float* mDeviceBias = nullptr; // GPU 端指针 (用于计算)
        };

        class MyGemmPluginCreator : public IPluginCreator
        {
        public:
            MyGemmPluginCreator();
            const char* getPluginName() const noexcept override;
            const char* getPluginVersion() const noexcept override;
            const PluginFieldCollection* getFieldNames() noexcept override;
            IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
            IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
            void setPluginNamespace(const char* pluginNamespace) noexcept override;
            const char* getPluginNamespace() const noexcept override;

        private:
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
            std::string mNamespace;
        };
    }
}