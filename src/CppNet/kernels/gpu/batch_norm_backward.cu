/**
 * @file batch_norm_backward.cu
 * @brief CUDA kernel for Batch Normalization backward pass
 *
 * Each block handles one feature. Uses shared memory reductions
 * to compute grad_gamma, grad_beta, dvar, and dmean.
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void batch_norm_backward_kernel(
                const float* __restrict__ grad_output,
                const float* __restrict__ input,
                const float* __restrict__ x_hat,
                const float* __restrict__ gamma,
                const float* __restrict__ batch_mean,
                const float* __restrict__ batch_var,
                float* __restrict__ grad_input,
                float* __restrict__ grad_gamma,
                float* __restrict__ grad_beta,
                int batch, int features, float eps)
            {
                int f = blockIdx.x;
                if (f >= features) return;

                extern __shared__ float smem[];
                float* s_dgamma = smem;
                float* s_dbeta = smem + blockDim.x;
                float* s_dvar = smem + 2 * blockDim.x;
                float* s_dmean = smem + 3 * blockDim.x;

                int tid = threadIdx.x;
                int blockSize = blockDim.x;

                float inv_std = rsqrtf(batch_var[f] + eps);
                float inv_batch = 1.0f / static_cast<float>(batch);
                float mean = batch_mean[f];
                float g = gamma[f];

                // Phase 1: accumulate grad_gamma, grad_beta, dvar, dmean
                float local_dgamma = 0.0f;
                float local_dbeta = 0.0f;
                float local_dvar = 0.0f;
                float local_dmean = 0.0f;

                for (int n = tid; n < batch; n += blockSize)
                {
                    int idx = n * features + f;
                    float go = grad_output[idx];
                    float xh = x_hat[idx];
                    float x_mu = input[idx] - mean;
                    float dx_hat = go * g;

                    local_dgamma += go * xh;
                    local_dbeta += go;
                    local_dvar += dx_hat * x_mu * (-0.5f) * inv_std * inv_std * inv_std;
                    local_dmean += dx_hat * (-inv_std);
                }

                s_dgamma[tid] = local_dgamma;
                s_dbeta[tid] = local_dbeta;
                s_dvar[tid] = local_dvar;
                s_dmean[tid] = local_dmean;
                __syncthreads();

                for (int s = blockSize / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                    {
                        s_dgamma[tid] += s_dgamma[tid + s];
                        s_dbeta[tid] += s_dbeta[tid + s];
                        s_dvar[tid] += s_dvar[tid + s];
                        s_dmean[tid] += s_dmean[tid + s];
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    grad_gamma[f] = s_dgamma[0];
                    grad_beta[f] = s_dbeta[0];
                }
                __syncthreads();

                float dvar = s_dvar[0];
                float dmean_val = s_dmean[0];

                // Phase 2: compute grad_input
                for (int n = tid; n < batch; n += blockSize)
                {
                    int idx = n * features + f;
                    float dx_hat = grad_output[idx] * g;
                    float x_mu = input[idx] - mean;

                    grad_input[idx] = dx_hat * inv_std +
                                      dvar * 2.0f * x_mu * inv_batch +
                                      dmean_val * inv_batch;
                }
            }
        }
    }
}
