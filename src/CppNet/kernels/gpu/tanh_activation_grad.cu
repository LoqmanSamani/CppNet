#include <cuda_runtime.h>



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            __global__ void tanh_grad_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int total_elements)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < total_elements)
                {
                    float t = Z[idx];
                    dZ[idx] = dA[idx] * (1.0f - t * t);
                }
            }
        } 
    } 
} 
