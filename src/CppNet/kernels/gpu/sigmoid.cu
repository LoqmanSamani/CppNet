#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {

            __global__ void sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int total_elements)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < total_elements)
                {
                    float x = input[idx];
                    float clamped = fmaxf(-500.0f, fminf(500.0f, x));
                    output[idx] = 1.0f / (1.0f + expf(-clamped));
                }
            }
        }
    }
}
