/**
 * @file global_max_pool2d.cu
 * @brief CUDA kernel for Global Max Pooling 2D
 *
 * Each thread handles one (batch, channel) pair.
 * Forward:  output[n,c] = max over H×W of input[n,c,h,w]; stores argmax
 * Backward: gradient flows only to the argmax position
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void global_max_pool2d_forward_kernel(
                const float* __restrict__ input,
                float* __restrict__ output,
                int* __restrict__ argmax_indices,
                int batch, int channels, int height, int width)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total = batch * channels;
                if (idx >= total) return;

                int n = idx / channels;
                int c = idx % channels;

                int spatial = height * width;
                int base = (n * channels + c) * spatial;

                float max_val = input[base];
                int max_idx = 0;

                for (int i = 1; i < spatial; ++i)
                {
                    float val = input[base + i];
                    if (val > max_val)
                    {
                        max_val = val;
                        max_idx = i;
                    }
                }

                output[idx] = max_val;
                argmax_indices[idx] = max_idx;
            }

            __global__ void global_max_pool2d_backward_kernel(
                const float* __restrict__ grad_output,
                const int* __restrict__ argmax_indices,
                float* __restrict__ grad_input,
                int batch, int channels, int spatial)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total = batch * channels;
                if (idx >= total) return;

                int base = idx * spatial;
                int max_pos = argmax_indices[idx];

                // grad_input was already zeroed; set only the argmax position
                grad_input[base + max_pos] = grad_output[idx];
            }
        }
    }
}
