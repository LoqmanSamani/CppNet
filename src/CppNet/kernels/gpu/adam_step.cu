#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__ void adam_step_kernel(float* W, const float* dW,
                                             float* m, float* v,
                                             float lr, float beta1,
                                             float beta2, float eps,
                                             float bc1, float bc2, int N)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < N)
                {
                    m[idx] = beta1 * m[idx] + (1.0f - beta1) * dW[idx];
                    v[idx] = beta2 * v[idx] + (1.0f - beta2) * dW[idx] * dW[idx];
                    float mh = m[idx] / bc1;
                    float vh = v[idx] / bc2;
                    W[idx] -= lr * mh / (sqrtf(vh) + eps);
                }
            }
        }
    }
}
