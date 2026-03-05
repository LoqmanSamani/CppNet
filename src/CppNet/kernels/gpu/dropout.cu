/**
 * @file dropout_forward.cu
 * @brief CUDA kernel for Dropout forward pass
 *
 * Applies a pre-generated mask and scales the output by 1/(1-p).
 * The mask is generated on CPU and transferred to GPU.
 */

#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void dropout_forward_kernel(
                const float* __restrict__ input,
                const float* __restrict__ mask,
                float* __restrict__ output,
                int total_elements, float scale)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total_elements)
                    output[idx] = input[idx] * mask[idx] * scale;
            }

            __global__ void dropout_backward_kernel(
                const float* __restrict__ grad_output,
                const float* __restrict__ mask,
                float* __restrict__ grad_input,
                int total_elements, float scale)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total_elements)
                    grad_input[idx] = grad_output[idx] * mask[idx] * scale;
            }
        }
    }
}
