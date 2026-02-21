#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"



namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void sgd_step_kernel(float* W, const float* dW, float LR, int TP)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < TP)
                {
                    W[idx] -= LR * dW[idx];
                }

            }

        }
    }
}
