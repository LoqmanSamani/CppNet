/**
 * @file huber.cu
 * @brief CUDA kernels for Huber (Smooth L1) loss
 *
 * Forward:  loss += 0.5*d^2 if |d|<=delta, else delta*(|d|-0.5*delta)
 * Backward: grad = scale*d   if |d|<=delta, else scale*delta*sign(d)
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void huber_forward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ target,
                float* __restrict__ loss,
                int total, float delta)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                {
                    float diff = pred[idx] - target[idx];
                    float abs_diff = fabsf(diff);
                    float val = (abs_diff <= delta)
                        ? 0.5f * diff * diff
                        : delta * (abs_diff - 0.5f * delta);
                    atomicAdd(loss, val);
                }
            }

            __global__ void huber_backward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ target,
                float* __restrict__ grad,
                int total, float scale, float delta)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                {
                    float diff = pred[idx] - target[idx];
                    float abs_diff = fabsf(diff);
                    if (abs_diff <= delta)
                        grad[idx] = scale * diff;
                    else
                        grad[idx] = scale * delta * ((diff > 0.0f) ? 1.0f : -1.0f);
                }
            }
        }
    }
}
