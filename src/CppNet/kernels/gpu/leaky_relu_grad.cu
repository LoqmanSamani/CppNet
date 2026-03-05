#include <cuda_runtime.h>



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            __global__ void leaky_relu_grad_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int total_elements, float alpha)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < total_elements)
                {
                    dZ[idx] = (Z[idx] > 0.0f) ? dA[idx] : alpha * dA[idx];
                }
            }
        } 
    } 
} 
