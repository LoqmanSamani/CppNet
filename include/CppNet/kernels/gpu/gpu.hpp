#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {

            __global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K);

            __global__ void add_bias_kernel(float* output, const float* bias, int M, int N);

            __global__ void matmul_grad_weights_kernel(const float* X, const float* dY, float* dW, int batch, int in_size, int out_size);

            __global__ void bias_grad_kernel(const float* dY, float* db, int batch, int out);

            __global__ void matmul_grad_input_kernel(const float* dY, const float* W, float* dX, int batch, int in_size, int out_size);

            __global__ void elementwise_kernel(float* A, float* B, float* C, int M, int N, int K);

            __global__ void sgd_step_kernel(float* W, const float* dW, int LR, int TP);

           __global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int total_elements);

            __global__ void relu_grad_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int total_elements);

        }
    }
}

#endif // USE_CUDA