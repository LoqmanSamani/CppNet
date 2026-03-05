#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void rmsprop_step_kernel(float* W, const float* dW,
                                                 float* cache, float lr,
                                                 float rho, float eps, int N)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < N)
                {
                    cache[idx] = rho * cache[idx] + (1.0f - rho) * dW[idx] * dW[idx];
                    W[idx] -= lr * dW[idx] / (sqrtf(cache[idx]) + eps);
                }
            }
        }
    }
}
