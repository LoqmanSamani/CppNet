/**
 * @file global_avg_pool2d.cu
 * @brief CUDA kernel for Global Average Pooling 2D
 *
 * Each thread handles one (batch, channel) pair.
 * Forward:  output[n,c] = mean over H×W of input[n,c,h,w]
 * Backward: grad_input[n,c,h,w] = grad_output[n,c] / (H*W)
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void global_avg_pool2d_forward_kernel(
                const float* __restrict__ input,
                float* __restrict__ output,
                int batch, int channels, int height, int width)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total = batch * channels;
                if (idx >= total) return;

                int n = idx / channels;
                int c = idx % channels;

                int spatial = height * width;
                float inv_spatial = 1.0f / static_cast<float>(spatial);

                float sum = 0.0f;
                int base = (n * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i)
                    sum += input[base + i];

                output[idx] = sum * inv_spatial;
            }

            __global__ void global_avg_pool2d_backward_kernel(
                const float* __restrict__ grad_output,
                float* __restrict__ grad_input,
                int batch, int channels, int height, int width)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int spatial = height * width;
                int total = batch * channels * spatial;
                if (idx >= total) return;

                int nc = idx / spatial;
                float inv_spatial = 1.0f / static_cast<float>(spatial);

                grad_input[idx] = grad_output[nc] * inv_spatial;
            }
        }
    }
}
