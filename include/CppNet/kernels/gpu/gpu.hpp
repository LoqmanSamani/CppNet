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

            void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K);

            void add_bias_gpu(float* output, const float* bias, int M, int N);

            void matmul_grad_weights_gpu(const float* X, const float* dY, float* dW, int batch, int in_size, int out_size);

            void bias_grad_gpu(const float* dY, float* db, int batch, int out);

            void matmul_grad_input_gpu(const float* dY, const float* W, float* dX, int batch, int in_size, int out_size);

            void elementwise_kernel(float* A, float* B, float* C, int M, int N, int K);

        }
    }
}

#endif // USE_CUDA