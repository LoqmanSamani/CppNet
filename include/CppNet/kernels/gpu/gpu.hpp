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

            __global__ void elementwise_kernel(float* A, float* B, float* C, int N, int op, int unused);

            __global__ void sgd_step_kernel(float* W, const float* dW, float LR, int TP);

           __global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int total_elements);

            __global__ void relu_grad_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int total_elements);

            // ── Conv2D kernels ──────────────────────────────────────
            __global__ void conv2d_forward_kernel(
                const float* input, const float* weights, const float* bias,
                float* output,
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride, int padding,
                int H_out, int W_out,
                bool use_bias);

            __global__ void conv2d_backward_kernel(
                const float* grad_output, const float* input_cache,
                const float* weights,
                float* grad_input, float* grad_weights, float* grad_biases,
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride, int padding,
                int H_out, int W_out,
                bool use_bias);

            // ── MaxPool2D kernels ───────────────────────────────────
            __global__ void maxpool2d_forward_kernel(
                const float* input, float* output, int* max_indices,
                int N, int C, int H, int W,
                int pool_size, int stride,
                int H_out, int W_out);

            __global__ void maxpool2d_backward_kernel(
                const float* grad_output, const int* max_indices,
                float* grad_input,
                int N, int C, int H, int W,
                int pool_size, int stride,
                int H_out, int W_out);

        }
    }
}

#endif // USE_CUDA