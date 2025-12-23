#include <iostream>
#include <vector>
#include <NvInfer.h>
#include <Windows.h> // 专门用来加载 DLL

using namespace nvinfer1;

// 简单的日志记录器
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印警告和错误
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

int main() {
    std::cout << ">>> Start Testing..." << std::endl;

    // ---------------------------------------------------------
    // 步骤 1: 加载插件 DLL
    // ---------------------------------------------------------
    // 这一步非常关键！如果不加载，REGISTER_TENSORRT_PLUGIN 宏就不会执行
    // 也就不会注册到 TensorRT 里
    HMODULE h = LoadLibraryA("MyGemmPlugin.dll");
    if (!h) {
        std::cerr << "!!! Failed to load MyGemmPlugin.dll. Check if the file exists." << std::endl;
        return -1;
    }
    std::cout << ">>> MyGemmPlugin.dll loaded successfully." << std::endl;

    // ---------------------------------------------------------
    // 步骤 2: 检查注册表 (Registry)
    // ---------------------------------------------------------
    // 获取全局插件注册表
    auto* registry = getPluginRegistry();

    // 查找我们的插件 Creator
    // "MyGemmPlugin" 是你在 Creator::getPluginName() 里写的名字
    // "1" 是版本号
    auto* creator = registry->getPluginCreator("MyGemmPlugin", "1");

    if (creator) {
        std::cout << ">>> SUCCESS: Found 'MyGemmPlugin' in TensorRT Registry!" << std::endl;
        std::cout << "    Plugin Name: " << creator->getPluginName() << std::endl;
        std::cout << "    Plugin Version: " << creator->getPluginVersion() << std::endl;
        std::cout << "    Plugin Namespace: " << creator->getPluginNamespace() << std::endl;
    }
    else {
        std::cout << "!!! FAILED: Could not find 'MyGemmPlugin' in Registry." << std::endl;

        // 打印出所有已注册的插件看看
        int numCreators = 0;
        auto* creatorList = registry->getPluginCreatorList(&numCreators);
        std::cout << "    Available plugins: ";
        for (int i = 0; i < numCreators; ++i) {
            std::cout << creatorList[i]->getPluginName() << " ";
        }
        std::cout << std::endl;
    }

    // 释放 DLL
    FreeLibrary(h);
    return 0;
}