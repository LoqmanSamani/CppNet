#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"



namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void sgd_step_kernel(float* W, float* dW, int LR, int TP)
            {
               
                int idx = blockIdx.x * blockIdx.x + threadIdx.x;
                if (idx < TP)
                {
                    W[idx] -= LR * dW[idx];
                }

            }

        }
    }
}
