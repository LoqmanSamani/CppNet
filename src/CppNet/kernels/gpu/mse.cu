/**
 * @file mse.cu
 * @brief CUDA kernels for Mean Squared Error loss
 *
 * Forward:  loss += (pred - target)^2  per element (atomicAdd reduction)
 * Backward: grad = scale * 2 * (pred - target)  per element
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void mse_forward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ target,
                float* __restrict__ loss,
                int total)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                {
                    float diff = pred[idx] - target[idx];
                    atomicAdd(loss, diff * diff);
                }
            }

            __global__ void mse_backward_kernel(
                const float* __restrict__ pred,
                const float* __restrict__ target,
                float* __restrict__ grad,
                int total, float scale)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total)
                    grad[idx] = scale * (pred[idx] - target[idx]);
            }
        }
    }
}
