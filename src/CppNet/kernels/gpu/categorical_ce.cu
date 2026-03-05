/**
 * @file categorical_ce.cu
 * @brief CUDA kernels for Categorical Cross-Entropy loss
 *
 * Two forward paths:
 *   from_logits=true  → softmax + CE (one block per row, shared-memory reduction)
 *   from_logits=false → -target * log(clipped_pred) per element + atomicAdd
 *
 * Two backward paths:
 *   from_logits=true  → grad = (softmax - target) * scale   (element-wise)
 *   from_logits=false → grad = -target / clipped_pred * scale (element-wise)
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            // ---- from_logits=true (one block per row) ----
            __global__ void categorical_ce_logits_forward_kernel(
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

                // find row max
                float local_max = -1e30f;
                for (int c = tid; c < classes; c += blockDim.x)
                    local_max = fmaxf(local_max, logits[offset + c]);
                smem[tid] = local_max;
                __syncthreads();
                for (int s = blockDim.x / 2; s > 0; s >>= 1)
                {
                    if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
                    __syncthreads();
                }
                float row_max = smem[0];
                __syncthreads();

                // exp + sum
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
                    if (tid < s) smem[tid] += smem[tid + s];
                    __syncthreads();
                }
                float sum_exp = smem[0];
                __syncthreads();

                // normalize + loss
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
                    if (tid < s) smem[tid] += smem[tid + s];
                    __syncthreads();
                }
                if (tid == 0) atomicAdd(loss, smem[0]);
            }

            __global__ void categorical_ce_logits_backward_kernel(
                const float* __restrict__ softmax,
                const float* __restrict__ targets,
                float* __restrict__ grad,
                int total, float scale)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                    grad[idx] = (softmax[idx] - targets[idx]) * scale;
            }

            // ---- from_logits=false (element-wise) ----
            __global__ void categorical_ce_probs_forward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ targets,
                float* __restrict__ loss,
                int total)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                {
                    float p = fmaxf(1e-15f, fminf(1.0f - 1e-15f, pred[idx]));
                    atomicAdd(loss, -targets[idx] * logf(p));
                }
            }

            __global__ void categorical_ce_probs_backward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ targets,
                float* __restrict__ grad,
                int total, float scale)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                {
                    float p = fmaxf(1e-15f, fminf(1.0f - 1e-15f, pred[idx]));
                    grad[idx] = -targets[idx] / p * scale;
                }
            }
        }
    }
}
