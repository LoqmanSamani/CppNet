/**
 * @file bce.cu
 * @brief CUDA kernels for Binary Cross-Entropy loss
 *
 * Forward:  loss += -[t*log(p) + (1-t)*log(1-p)]  per element
 * Backward: grad = scale * (p - t) / (p * (1 - p))  per element
 *           with clipping to [1e-7, 1-1e-7]
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void bce_forward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ target,
                float* __restrict__ loss,
                int total)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                {
                    float p = fmaxf(1e-7f, fminf(pred[idx], 1.0f - 1e-7f));
                    float t = target[idx];
                    float val = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
                    atomicAdd(loss, val);
                }
            }

            __global__ void bce_backward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ target,
                float* __restrict__ grad,
                int total, float scale)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                {
                    float p = fmaxf(1e-7f, fminf(pred[idx], 1.0f - 1e-7f));
                    float t = target[idx];
                    grad[idx] = scale * (p - t) / (p * (1.0f - p));
                }
            }
        }
    }
}
