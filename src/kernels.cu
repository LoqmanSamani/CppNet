#include "kernels.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul_kernel(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

namespace CppNet {
    void matmul_gpu(const float *A, const float *B, float *C, int M, int N, int K) {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        
        matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel execution error: " << cudaGetErrorString(err) << std::endl;
        }
    }
}
