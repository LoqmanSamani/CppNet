#include <cuda_runtime.h>
#include "CppNet/kernels/gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            __global__void grad_weights_kernel(const float* X, const float* dY, float* dW, int batch, int in_size, int out_size)
            {
                int i = blockIdx.y * blockDim.y + threadIdx.y;  // in_size
                int j = blockIdx.x * blockDim.x + threadIdx.x;  // out_size

                if (i < in_size && j < out_size)
                {
                    float sum = 0.f;
                    for (int b = 0; b < batch; b++)
                    {
                        sum += X[b * in_size + i] * dY[b * out_size + j];
                    }
                    dW[i * out_size + j] = sum;
                }
            }

            void matmul_grad_weights_gpu(const float* X, const float* dY, float* dW, int batch, int in_size, int out_size)
            {
                float *dX, *ddY, *ddW;

                cudaMalloc(&dX, batch * in_size * sizeof(float));
                cudaMalloc(&ddY, batch * out_size * sizeof(float));
                cudaMalloc(&ddW, in_size * out_size * sizeof(float));

                cudaMemcpy(dX, X, batch * in_size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(ddY, dY, batch * out_size * sizeof(float), cudaMemcpyHostToDevice);

                dim3 block(16, 16);
                dim3 grid((out_size + 15)/16, (in_size + 15)/16);

                grad_weights_kernel<<<grid, block>>>(dX, ddY, ddW, batch, in_size, out_size);

                cudaMemcpy(dW, ddW, in_size * out_size * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(dX);
                cudaFree(ddY);
                cudaFree(ddW);
            }

        } 
    } 
}
