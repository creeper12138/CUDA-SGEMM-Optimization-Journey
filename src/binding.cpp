#include <torch/extension.h>
#include <vector>

// 声明我们在 .cu 文件里写的那个封装函数
// extern "C" 防止 C++ 编译器把函数名改乱了 (Name Mangling)
void launch_matrix_mul(float* A, float* B, float* C, int M, int N, int K);

// 这是 Python 会调用的函数
torch::Tensor matrix_mul_cuda(torch::Tensor a, torch::Tensor b) {
    // 1. 检查输入
    // 必须是 CUDA Tensor，必须是连续内存，必须是 float32
    TORCH_CHECK(a.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "B must be float32");

    // 2. 获取维度
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    // 检查矩阵乘法维度规则 (M x K) * (K x N)
    TORCH_CHECK(b.size(0) == K, "Shape mismatch: A(MxK) and B(KxN)");

    // 3. 创建输出 Tensor (在 GPU 上)
    auto c = torch::zeros({ M, N }, a.options());

    // 4. 获取数据指针 (这就是 C++ 指针！)
    float* A_ptr = a.data_ptr<float>();
    float* B_ptr = b.data_ptr<float>();
    float* C_ptr = c.data_ptr<float>();

    // 5. 调用 CUDA Kernel
    launch_matrix_mul(A_ptr, B_ptr, C_ptr, M, N, K);

    return c;
}

// 6. 绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matrix_mul_cuda, "Matrix Multiplication (CUDA)");
}