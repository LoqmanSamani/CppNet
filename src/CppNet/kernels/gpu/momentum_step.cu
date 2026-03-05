#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void momentum_step_kernel(float* W, const float* dW,
                                                  float* vel, float mu,
                                                  float lr, int N)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < N)
                {
                    vel[idx] = mu * vel[idx] - lr * dW[idx];
                    W[idx] += vel[idx];
                }
            }
        }
    }
}
