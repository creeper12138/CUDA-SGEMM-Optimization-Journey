#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include <cublas_v2.h> // <--- 新增这个

// =================================================================================
// 工具宏与辅助函数
// =================================================================================
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

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

// =================================================================================
// Phase 0: Naive (基准线)
// 修正点：修复了 n -> N 的拼写错误
// =================================================================================
__global__ void sgemm_0_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum; // 这里之前写成了 n，已修正为 N
    }
}

// =================================================================================
// Phase 1: Shared Memory (分块优化)
// =================================================================================
template <int BLOCK_SIZE>
__global__ void sgemm_1_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // 加载数据
        if (row < M && (k + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (k + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 计算
        for (int i = 0; i < BLOCK_SIZE; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// =================================================================================
// Phase 2: Vectorized (float4 访存)
// =================================================================================
template <int BLOCK_SIZE>
__global__ void sgemm_2_vectorized(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 线程映射：BLOCK_SIZE 32x32 = 1024 个元素
    // 每个线程搬运 float4 (4个float)，所以只需要 256 个线程参与搬运
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int load_row = tid / (BLOCK_SIZE / 4);
    int load_col = (tid % (BLOCK_SIZE / 4)) * 4;

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // 只有前256个线程干活搬运 A (假设 BLOCK_SIZE=32)
        if (tid < (BLOCK_SIZE * BLOCK_SIZE / 4)) {
            // 这里为了安全，应当加边界检查，但为了性能演示且N是32倍数，先略过严格边界
            float4 v_a = reinterpret_cast<const float4*>(&A[(blockIdx.y * BLOCK_SIZE + load_row) * K + (k + load_col)])[0];
            As[load_row][load_col + 0] = v_a.x; As[load_row][load_col + 1] = v_a.y;
            As[load_row][load_col + 2] = v_a.z; As[load_row][load_col + 3] = v_a.w;

            float4 v_b = reinterpret_cast<const float4*>(&B[(k + load_row) * N + (blockIdx.x * BLOCK_SIZE + load_col)])[0];
            Bs[load_row][load_col + 0] = v_b.x; Bs[load_row][load_col + 1] = v_b.y;
            Bs[load_row][load_col + 2] = v_b.z; Bs[load_row][load_col + 3] = v_b.w;
        }
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// =================================================================================
// 核心参数定义 (Phase 3, 4, 5 通用)
// =================================================================================
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// =================================================================================
// Phase 3: Register Tiling (单缓冲 Single Buffer)
// =================================================================================
__global__ void sgemm_3_reg_tiled_1buf(const float* A, const float* B, float* C, int M, int N, int K) {
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;

    float thread_results[TM][TN] = { 0.0f };
    float reg_a[TM], reg_b[TN];

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const float* A_ptr = A + by * BM * K;
    const float* B_ptr = B + bx * BN;

    for (int k = 0; k < K; k += BK) {
        // --- 搬运 A (128x8) ---
        int load_a_row = tid / 2;
        int load_a_col = (tid % 2) * 4;
        if (load_a_row < BM) {
            float4 v = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + k + load_a_col])[0];
            As[load_a_row][load_a_col] = v.x; As[load_a_row][load_a_col + 1] = v.y;
            As[load_a_row][load_a_col + 2] = v.z; As[load_a_row][load_a_col + 3] = v.w;
        }
        // --- 搬运 B (8x128) ---
        int load_b_row = tid / 32;
        int load_b_col = (tid % 32) * 4;
        if (load_b_row < BK) {
            float4 v = reinterpret_cast<const float4*>(&B_ptr[(k + load_b_row) * N + load_b_col])[0];
            Bs[load_b_row][load_b_col] = v.x; Bs[load_b_row][load_b_col + 1] = v.y;
            Bs[load_b_row][load_b_col + 2] = v.z; Bs[load_b_row][load_b_col + 3] = v.w;
        }
        __syncthreads();

        // --- 计算 ---
#pragma unroll
        for (int i = 0; i < BK; ++i) {
#pragma unroll
            for (int r = 0; r < TM; ++r) reg_a[r] = As[ty * TM + r][i];
#pragma unroll
            for (int c = 0; c < TN; ++c) reg_b[c] = Bs[i][tx * TN + c];
#pragma unroll
            for (int r = 0; r < TM; ++r)
                for (int c = 0; c < TN; ++c) thread_results[r][c] += reg_a[r] * reg_b[c];
        }
        __syncthreads();
    }

    // 写回
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        for (int c = 0; c < TN; ++c) {
            int global_row = by * BM + ty * TM + r;
            int global_col = bx * BN + tx * TN + c;
            if (global_row < M && global_col < N)
                C[global_row * N + global_col] = thread_results[r][c];
        }
    }
}

// =================================================================================
// Phase 4: Double Buffering (软件双缓冲)
// =================================================================================
__global__ void sgemm_4_double_buffer(const float* A, const float* B, float* C, int M, int N, int K) {
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;

    float thread_results[TM][TN] = { 0.0f };
    float reg_a[TM], reg_b[TN];

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    float4 load_a_reg, load_b_reg;

    const float* A_ptr = A + by * BM * K;
    const float* B_ptr = B + bx * BN;

    int write_stage = 1;
    int read_stage = 0;

    // --- Prologue ---
    {
        // 预取第0块到 Shared[0]
        int load_a_row = tid / 2; int load_a_col = (tid % 2) * 4;
        if (load_a_row < BM) {
            load_a_reg = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + 0 + load_a_col])[0];
            As[0][load_a_row][load_a_col] = load_a_reg.x; As[0][load_a_row][load_a_col + 1] = load_a_reg.y;
            As[0][load_a_row][load_a_col + 2] = load_a_reg.z; As[0][load_a_row][load_a_col + 3] = load_a_reg.w;
        }
        int load_b_row = tid / 32; int load_b_col = (tid % 32) * 4;
        if (load_b_row < BK) {
            load_b_reg = reinterpret_cast<const float4*>(&B_ptr[(0 + load_b_row) * N + load_b_col])[0];
            Bs[0][load_b_row][load_b_col] = load_b_reg.x; Bs[0][load_b_row][load_b_col + 1] = load_b_reg.y;
            Bs[0][load_b_row][load_b_col + 2] = load_b_reg.z; Bs[0][load_b_row][load_b_col + 3] = load_b_reg.w;
        }
    }
    __syncthreads();

    // --- Main Loop ---
    for (int k = 0; k < K; k += BK) {
        int next_k = k + BK;

        // 1. 预取下一块到寄存器
        if (next_k < K) {
            int load_a_row = tid / 2; int load_a_col = (tid % 2) * 4;
            if (load_a_row < BM)
                load_a_reg = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + next_k + load_a_col])[0];

            int load_b_row = tid / 32; int load_b_col = (tid % 32) * 4;
            if (load_b_row < BK)
                load_b_reg = reinterpret_cast<const float4*>(&B_ptr[(next_k + load_b_row) * N + load_b_col])[0];
        }

        // 2. 计算当前块 Shared[read_stage]
#pragma unroll
        for (int i = 0; i < BK; ++i) {
#pragma unroll
            for (int r = 0; r < TM; ++r) reg_a[r] = As[read_stage][ty * TM + r][i];
#pragma unroll
            for (int c = 0; c < TN; ++c) reg_b[c] = Bs[read_stage][i][tx * TN + c];
#pragma unroll
            for (int r = 0; r < TM; ++r)
                for (int c = 0; c < TN; ++c) thread_results[r][c] += reg_a[r] * reg_b[c];
        }

        // 3. 填入 Shared[write_stage]
        __syncthreads(); // 必须等待大家读完 read_stage

        if (next_k < K) {
            int load_a_row = tid / 2; int load_a_col = (tid % 2) * 4;
            if (load_a_row < BM) {
                As[write_stage][load_a_row][load_a_col] = load_a_reg.x; As[write_stage][load_a_row][load_a_col + 1] = load_a_reg.y;
                As[write_stage][load_a_row][load_a_col + 2] = load_a_reg.z; As[write_stage][load_a_row][load_a_col + 3] = load_a_reg.w;
            }
            int load_b_row = tid / 32; int load_b_col = (tid % 32) * 4;
            if (load_b_row < BK) {
                Bs[write_stage][load_b_row][load_b_col] = load_b_reg.x; Bs[write_stage][load_b_row][load_b_col + 1] = load_b_reg.y;
                Bs[write_stage][load_b_row][load_b_col + 2] = load_b_reg.z; Bs[write_stage][load_b_row][load_b_col + 3] = load_b_reg.w;
            }
            read_stage ^= 1;
            write_stage ^= 1;
        }
        __syncthreads(); // 必须等待大家写完 write_stage
    }

    // 写回
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        for (int c = 0; c < TN; ++c) {
            int global_row = by * BM + ty * TM + r;
            int global_col = bx * BN + tx * TN + c;
            if (global_row < M && global_col < N)
                C[global_row * N + global_col] = thread_results[r][c];
        }
    }
}

// =================================================================================
// Phase 5: Async Copy (Ampere异步流水线)
// =================================================================================
__global__ void sgemm_5_async(float* A, float* B, float* C, int M, int N, int K) {
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;

    float thread_results[TM][TN] = { 0.0f };
    float reg_a[TM], reg_b[TN];

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    float* A_ptr = A + by * BM * K;
    float* B_ptr = B + bx * BN;
    int write_stage = 1, read_stage = 0;

    // --- Prologue (Async) ---
    int load_a_row = tid / 2; int load_a_col = (tid % 2) * 4;
    int load_b_row = tid / 32; int load_b_col = (tid % 32) * 4;

    // 异步加载第0块
    __pipeline_memcpy_async(&As[0][load_a_row][load_a_col], &A_ptr[load_a_row * K + 0 + load_a_col], 16);
    __pipeline_memcpy_async(&Bs[0][load_b_row][load_b_col], &B_ptr[(0 + load_b_row) * N + load_b_col], 16);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // --- Main Loop ---
    for (int k = 0; k < K; k += BK) {
        int next_k = k + BK;

        // 1. 异步搬运下一块
        if (next_k < K) {
            __pipeline_memcpy_async(&As[write_stage][load_a_row][load_a_col], &A_ptr[load_a_row * K + next_k + load_a_col], 16);
            __pipeline_memcpy_async(&Bs[write_stage][load_b_row][load_b_col], &B_ptr[(next_k + load_b_row) * N + load_b_col], 16);
        }
        __pipeline_commit(); // 提交任务

        // 2. 计算当前块
#pragma unroll
        for (int i = 0; i < BK; ++i) {
#pragma unroll
            for (int r = 0; r < TM; ++r) reg_a[r] = As[read_stage][ty * TM + r][i];
#pragma unroll
            for (int c = 0; c < TN; ++c) reg_b[c] = Bs[read_stage][i][tx * TN + c];
#pragma unroll
            for (int r = 0; r < TM; ++r)
                for (int c = 0; c < TN; ++c) thread_results[r][c] += reg_a[r] * reg_b[c];
        }

        // 3. 等待异步搬运
        __pipeline_wait_prior(0); // 等待刚才提交的那一组完成
        __syncthreads();

        if (next_k < K) {
            read_stage ^= 1;
            write_stage ^= 1;
        }
    }

#pragma unroll
    for (int r = 0; r < TM; ++r) {
        for (int c = 0; c < TN; ++c) {
            int global_row = by * BM + ty * TM + r;
            int global_col = bx * BN + tx * TN + c;
            if (global_row < M && global_col < N)
                C[global_row * N + global_col] = thread_results[r][c];
        }
    }
}

// =================================================================================
// Reference: cuBLAS (NVIDIA 官方库)
// =================================================================================
void bench_cublas(float* d_A, float* d_B, float* d_C, int M, int N, int K, int iter) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS 默认是列主序 (Column Major)，而C++是行主序 (Row Major)。
    // 为了利用 A*B = (B^T * A^T)^T 的数学性质，
    // 我们交换 A 和 B 的顺序传给 cuBLAS，这样算出来的 C 在内存里就是正确的行主序结果。
    // 不过对于跑分来说，怎么传都不影响速度，这里用最简单的方式：
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // 预热
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        // 注意这里为了适配行主序，我们逻辑上交换了 A 和 B
        // 实际计算的是 C = A * B
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 计算性能
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (ops * 1e-12) / (ms / iter / 1000.0f);

    printf("%-25s | Time: %7.3f ms | Perf: %6.3f TFLOPS\n", "Reference: cuBLAS", ms / iter, tflops);

    cublasDestroy(handle);
}


// =================================================================================
// Main Harness (评测主程序)
// =================================================================================
void print_perf(const char* name, float ms, double ops) {
    double tflops = (ops * 1e-12) / (ms / 1000.0f);
    printf("%-25s | Time: %7.3f ms | Perf: %6.3f TFLOPS\n", name, ms, tflops);
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    size_t size = M * N * sizeof(float);
    double ops = 2.0 * (double)M * (double)N * (double)K;

    printf("\nOptimization Journey: From Zero to Hero\n");
    printf("Matrix Size: %d x %d x %d\n", M, N, K);
    printf("====================================================================\n");

    // Host Memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);

    srand(0);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;

    // Device Memory
    float* d_A, * d_B, * d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size)); // 简化：这里我们让M=N=K，所以size一样
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms = 0;
    int iter = 5;

    // --------------------------------------------------------
    // Phase 0: Naive
    dim3 b32(32, 32); dim3 g32(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) sgemm_0_naive << <g32, b32 >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    print_perf("Phase 0: Naive", ms / iter, ops);

    // --------------------------------------------------------
    // Phase 1: Shared Memory
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) sgemm_1_shared<32> << <g32, b32 >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    print_perf("Phase 1: Shared Mem", ms / iter, ops);

    // --------------------------------------------------------
    // Phase 2: Vectorized
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) sgemm_2_vectorized<32> << <g32, b32 >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    print_perf("Phase 2: Vectorized", ms / iter, ops);

    // --------------------------------------------------------
    // Phase 3: Register Tiled (Single Buffer)
    // 注意：这里使用 16x16 的 Block (256 threads)
    dim3 b16(16, 16); dim3 g128(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) sgemm_3_reg_tiled_1buf << <g128, b16 >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    print_perf("Phase 3: Reg Tiled", ms / iter, ops);

    // --------------------------------------------------------
    // Phase 4: Double Buffer (Software)
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) sgemm_4_double_buffer << <g128, b16 >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    print_perf("Phase 4: Double Buffer", ms / iter, ops);

    // --------------------------------------------------------
    // Phase 5: Async Pipeline (Final)
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) sgemm_5_async << <g128, b16 >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    print_perf("Phase 5: Async Final", ms / iter, ops);

    printf("====================================================================\n");
    bench_cublas(d_A, d_B, d_C, M, N, K, iter); // <--- 新增这行

    printf("====================================================================\n");
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}