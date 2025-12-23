#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <cuda_pipeline.h>

// --- 核心参数 ---
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// --- Kernel 定义 ---
// 注意：参数列表里增加了 Bias 和 use_relu
__global__ void matrixMulPhase6_Async(
    float* A, float* B, float* C,
    const float* Bias,  // [新增] Bias 权重
    int use_relu,      // [新增] ReLU 开关
    int M, int N, int K)
{


    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;

    // [1] 寄存器与 Shared Memory
    float thread_results[TM][TN] = { 0.0f };
    float reg_a[TM];
    float reg_b[TN];

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    float* A_ptr = A + by * BM * K;
    float* B_ptr = B + bx * BN;

    int write_stage = 1;
    int read_stage = 0;

    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("[GPU Kernel] Received Bias Ptr: %p, use_relu: %d\n", Bias, use_relu);
    }


    // [2] Prologue: 预取第 0 块
    {
        int load_a_row = tid / 2;
        int load_a_col = (tid % 2) * 4;
        void* dest = &As[0][load_a_row][load_a_col];
        void* src = &A_ptr[load_a_row * K + 0 + load_a_col];
        __pipeline_memcpy_async(dest, src, 16);
    }
    {
        int load_b_row = tid / 32;
        int load_b_col = (tid % 32) * 4;
        void* dest = &Bs[0][load_b_row][load_b_col];
        void* src = &B_ptr[(0 + load_b_row) * N + load_b_col];
        __pipeline_memcpy_async(dest, src, 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // [3] Main Loop
    for (int k = 0; k < K; k += BK) {
        int next_k = k + BK;

        // 异步搬运下一块
        if (next_k < K) {
            int load_a_row = tid / 2;
            int load_a_col = (tid % 2) * 4;
            void* dest = &As[write_stage][load_a_row][load_a_col];
            void* src = &A_ptr[load_a_row * K + next_k + load_a_col];
            __pipeline_memcpy_async(dest, src, 16);

            int load_b_row = tid / 32;
            int load_b_col = (tid % 32) * 4;
            dest = &Bs[write_stage][load_b_row][load_b_col];
            src = &B_ptr[(next_k + load_b_row) * N + load_b_col];
            __pipeline_memcpy_async(dest, src, 16);
        }
        __pipeline_commit();

        // 计算当前块
        // 修正 warning: #pragma unroll 必须紧贴循环
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

        // 等待搬运完成
        __pipeline_wait_prior(0);
        __syncthreads();

        if (next_k < K) {
            read_stage ^= 1;
            write_stage ^= 1;
        }
    }

    // [4] 写回 Global Memory (包含融合逻辑)
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        for (int c = 0; c < TN; ++c) {
            int global_row = by * BM + ty * TM + r;
            int global_col = bx * BN + tx * TN + c;

            if (global_row < M && global_col < N) {
                float val = thread_results[r][c];

                // --- 融合逻辑 ---
                // 1. 加 Bias
                if (Bias != nullptr) {
                    val += Bias[global_col];
                }
                // 2. ReLU
                if (use_relu) {
                    val = (val > 0.0f) ? val : 0.0f;
                }
                // ---------------

                C[global_row * N + global_col] = val;
            }
        }
    }
}

// --- Launcher 定义 ---
// 给插件调用的接口
extern "C" void matrixMulPhase6_AsyncLauncher(
    float* A, float* B, float* C,
    const float* Bias, int use_relu, // <--- 改为 int
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 128 - 1) / 128, (M + 128 - 1) / 128);

    matrixMulPhase6_Async << <blocksPerGrid, threadsPerBlock, 0, stream >> > (
        A, B, C, Bias, use_relu, M, N, K
        );
}

// --- 主函数 (用于独立测试) ---
int main() {
    // 只是为了编译通过，内容为空即可
    // 如果你想测试，可以在这里写 Host 代码
    return 0;
}