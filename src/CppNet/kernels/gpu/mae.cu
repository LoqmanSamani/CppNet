/**
 * @file mae.cu
 * @brief CUDA kernels for Mean Absolute Error loss
 *
 * Forward:  loss += |pred - target|  per element (atomicAdd reduction)
 * Backward: grad = scale * sign(pred - target)  per element
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void mae_forward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ target,
                float* __restrict__ loss,
                int total)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                    atomicAdd(loss, fabsf(pred[idx] - target[idx]));
            }

            __global__ void mae_backward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ target,
                float* __restrict__ grad,
                int total, float scale)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                {
                    float diff = pred[idx] - target[idx];
                    grad[idx] = (diff > 0.0f) ? scale : ((diff < 0.0f) ? -scale : 0.0f);
                }
            }
        }
    }
}
