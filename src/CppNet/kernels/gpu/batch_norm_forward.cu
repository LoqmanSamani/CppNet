/**
 * @file batch_norm_forward.cu
 * @brief CUDA kernel for Batch Normalization forward pass
 *
 * Each block handles one feature. Threads within a block
 * cooperate via shared memory to compute mean and variance
 * across the batch dimension.
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void batch_norm_forward_kernel(
                const float* __restrict__ input,
                float* __restrict__ output,
                float* __restrict__ x_hat,
                const float* __restrict__ gamma,
                const float* __restrict__ beta,
                float* __restrict__ batch_mean,
                float* __restrict__ batch_var,
                float* __restrict__ running_mean,
                float* __restrict__ running_var,
                int batch, int features,
                float eps, float momentum, bool training)
            {
                int f = blockIdx.x;
                if (f >= features) return;

                extern __shared__ float smem[];

                int tid = threadIdx.x;
                int blockSize = blockDim.x;

                if (training)
                {
                    // Phase 1: compute mean
                    float local_sum = 0.0f;
                    for (int n = tid; n < batch; n += blockSize)
                        local_sum += input[n * features + f];

                    smem[tid] = local_sum;
                    __syncthreads();

                    for (int s = blockSize / 2; s > 0; s >>= 1)
                    {
                        if (tid < s)
                            smem[tid] += smem[tid + s];
                        __syncthreads();
                    }

                    float mean = smem[0] / static_cast<float>(batch);
                    if (tid == 0) batch_mean[f] = mean;
                    __syncthreads();

                    // Phase 2: compute variance
                    float local_var = 0.0f;
                    for (int n = tid; n < batch; n += blockSize)
                    {
                        float diff = input[n * features + f] - mean;
                        local_var += diff * diff;
                    }

                    smem[tid] = local_var;
                    __syncthreads();

                    for (int s = blockSize / 2; s > 0; s >>= 1)
                    {
                        if (tid < s)
                            smem[tid] += smem[tid + s];
                        __syncthreads();
                    }

                    float var = smem[0] / static_cast<float>(batch);
                    if (tid == 0) batch_var[f] = var;
                    __syncthreads();

                    float inv_std = rsqrtf(var + eps);

                    // Phase 3: normalize and scale
                    for (int n = tid; n < batch; n += blockSize)
                    {
                        int idx = n * features + f;
                        float xh = (input[idx] - mean) * inv_std;
                        x_hat[idx] = xh;
                        output[idx] = gamma[f] * xh + beta[f];
                    }

                    // Update running stats
                    if (tid == 0)
                    {
                        running_mean[f] = (1.0f - momentum) * running_mean[f] + momentum * mean;
                        running_var[f] = (1.0f - momentum) * running_var[f] + momentum * var;
                    }
                }
                else
                {
                    // Inference: use running statistics
                    float mean = running_mean[f];
                    float inv_std = rsqrtf(running_var[f] + eps);

                    for (int n = tid; n < batch; n += blockSize)
                    {
                        int idx = n * features + f;
                        float xh = (input[idx] - mean) * inv_std;
                        x_hat[idx] = xh;
                        output[idx] = gamma[f] * xh + beta[f];
                    }
                }
            }
        }
    }
}
