#include <cuda_runtime.h>
#include <iostream>
#include "CppNet/kernels/gpu/gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {

            __global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int total_elements)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < total_elements)
                {
                    float x = input[idx];
                    output[idx] = x > 0.0f ? x : 0.0f;
                }
            }
        }
    }
}