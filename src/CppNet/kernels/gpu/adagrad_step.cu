#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void adagrad_step_kernel(float* W, const float* dW,
                                                 float* accum, float lr,
                                                 float eps, int N)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < N)
                {
                    accum[idx] += dW[idx] * dW[idx];
                    W[idx] -= lr * dW[idx] / (sqrtf(accum[idx]) + eps);
                }
            }
        }
    }
}
