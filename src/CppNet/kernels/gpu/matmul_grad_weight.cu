#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            #define TILE_SIZE 32  // 32x32 tiles

            __global__ void matmul_grad_weights_kernel(
                const float* X,   // [batch x in_size]
                const float* dY,  // [batch x out_size]
                float* dW,        // [in_size x out_size]
                int batch, int in_size, int out_size)
            {
                
                int i = blockIdx.y * TILE_SIZE + threadIdx.y;  
                int j = blockIdx.x * TILE_SIZE + threadIdx.x; 
                
                __shared__ float tile_X[TILE_SIZE][TILE_SIZE];
                __shared__ float tile_dY[TILE_SIZE][TILE_SIZE];
                
                float sum = 0.0f;
                int num_tiles = (batch + TILE_SIZE - 1) / TILE_SIZE;
                
                for (int t = 0; t < num_tiles; t++) 
                {
                    // load tile of X^T (so we access X[b, i] as X_transposed[i, b])
                    int b = t * TILE_SIZE + threadIdx.x;
                    if (i < in_size && b < batch) 
                    {
                        tile_X[threadIdx.y][threadIdx.x] = X[b * in_size + i];
                    } 
                    else 
                    {
                        tile_X[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                    
                    // load tile of dY
                    b = t * TILE_SIZE + threadIdx.y;
                    if (b < batch && j < out_size) 
                    {
                        tile_dY[threadIdx.y][threadIdx.x] = dY[b * out_size + j];
                    } 
                    else 
                    {
                        tile_dY[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                    
                    __syncthreads();
                    
                    #pragma unroll
                    for (int k = 0; k < TILE_SIZE; k++) 
                    {
                        sum += tile_X[threadIdx.y][k] * tile_dY[k][threadIdx.x];
                    }
                    
                    __syncthreads();
                }
                
                if (i < in_size && j < out_size)
                {
                    atomicAdd(&dW[i * out_size + j], sum);
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

                matmul_grad_weights_kernel<<<grid, block>>>(dX, ddY, ddW, batch, in_size, out_size);

                cudaMemcpy(dW, ddW, in_size * out_size * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(dX);
                cudaFree(ddY);
                cudaFree(ddW);
            }

        } 
    } 
}
