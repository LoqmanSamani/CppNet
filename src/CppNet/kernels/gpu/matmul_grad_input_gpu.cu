#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/matmul_gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            __global__void grad_input_kernel(const float* dY, const float* W, float* dX, int batch, int in_size, int out_size)
            {
                int b = blockIdx.y * blockDim.y + threadIdx.y; // batch
                int i = blockIdx.x * blockDim.x + threadIdx.x; // in_size

                if (b < batch && i < in_size)
                {
                    float sum = 0.f;
                    for (int j = 0; j < out_size; j++)
                    {
                        sum += dY[b * out_size + j] * W[i * out_size + j];
                    }
                    dX[b * in_size + i] = sum;
                }
            }

            void matmul_grad_input_gpu(const float* dY, const float* W, float* dX, int batch, int in_size, int out_size)
            {
                float *ddY, *dW, *ddX;

                cudaMalloc(&ddY, batch * out_size * sizeof(float));
                cudaMalloc(&dW, in_size * out_size * sizeof(float));
                cudaMalloc(&ddX, batch * in_size * sizeof(float));

                cudaMemcpy(ddY, dY, batch * out_size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(dW, W, in_size * out_size * sizeof(float), cudaMemcpyHostToDevice);

                dim3 block(16, 16);
                dim3 grid((in_size + 15)/16, (batch + 15)/16);

                grad_input_kernel<<<grid, block>>>(ddY, dW, ddX, batch, in_size, out_size);

                cudaMemcpy(dX, ddX, batch * in_size * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(ddY);
                cudaFree(ddX);
                cudaFree(dW);
            }
        } 
    } 
}
