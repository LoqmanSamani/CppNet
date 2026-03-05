/**
 * @file softmax_ce.cu
 * @brief CUDA kernels for fused Softmax + Cross-Entropy loss
 *
 * Forward kernel: one block per batch row
 *   - numerically stable softmax (max-subtract trick)
 *   - stores softmax output for backward
 *   - computes -sum(target * log_softmax) per row, atomicAdds to global loss
 *
 * Backward kernel: element-wise  grad = (softmax - target) * scale
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void softmax_ce_forward_kernel(
                const float* __restrict__ logits,
                const float* __restrict__ targets,
                float* __restrict__ softmax_out,
                float* __restrict__ loss,
                int batch, int classes)
            {
                int row = blockIdx.x;
                if (row >= batch) return;

                extern __shared__ float smem[];
                int tid = threadIdx.x;
                int offset = row * classes;

                // Step 1: find row max
                float local_max = -1e30f;
                for (int c = tid; c < classes; c += blockDim.x)
                    local_max = fmaxf(local_max, logits[offset + c]);
                smem[tid] = local_max;
                __syncthreads();

                for (int s = blockDim.x / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                        smem[tid] = fmaxf(smem[tid], smem[tid + s]);
                    __syncthreads();
                }
                float row_max = smem[0];
                __syncthreads();

                // Step 2: exp and sum
                float local_sum = 0.0f;
                for (int c = tid; c < classes; c += blockDim.x)
                {
                    float e = expf(logits[offset + c] - row_max);
                    softmax_out[offset + c] = e;
                    local_sum += e;
                }
                smem[tid] = local_sum;
                __syncthreads();

                for (int s = blockDim.x / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                        smem[tid] += smem[tid + s];
                    __syncthreads();
                }
                float sum_exp = smem[0];
                __syncthreads();

                // Step 3: normalize softmax and compute loss
                float local_loss = 0.0f;
                float log_sum = logf(sum_exp + 1e-12f);
                for (int c = tid; c < classes; c += blockDim.x)
                {
                    softmax_out[offset + c] /= sum_exp;
                    float log_softmax = (logits[offset + c] - row_max) - log_sum;
                    local_loss -= targets[offset + c] * log_softmax;
                }

                smem[tid] = local_loss;
                __syncthreads();

                for (int s = blockDim.x / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                        smem[tid] += smem[tid + s];
                    __syncthreads();
                }
                if (tid == 0)
                    atomicAdd(loss, smem[0]);
            }

            __global__ void softmax_ce_backward_kernel(
                const float* __restrict__ softmax,
                const float* __restrict__ targets,
                float* __restrict__ grad,
                int total, float scale)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                    grad[idx] = (softmax[idx] - targets[idx]) * scale;
            }
        }
    }
}
