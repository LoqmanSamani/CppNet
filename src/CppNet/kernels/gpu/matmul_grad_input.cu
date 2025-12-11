#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            #define TILE_SIZE 32  // 32x32 tiles 

            __global__ void matmul_grad_input_kernel(
                const float* dY,  // [batch x out_size]
                const float* W,   // [in_size x out_size]
                float* dX,        // [batch x in_size]
                int batch, int in_size, int out_size)
            {
                int b = blockIdx.y * TILE_SIZE + threadIdx.y;
                int i = blockIdx.x * TILE_SIZE + threadIdx.x;
                
                __shared__ float tile_dY[TILE_SIZE][TILE_SIZE];
                __shared__ float tile_W[TILE_SIZE][TILE_SIZE];
                
                float sum = 0.0f;
                int num_tiles = (out_size + TILE_SIZE - 1) / TILE_SIZE;
                
                for (int t = 0; t < num_tiles; t++) 
                {
                    int j = t * TILE_SIZE + threadIdx.x;
                    if (b < batch && j < out_size) 
                    {
                        tile_dY[threadIdx.y][threadIdx.x] = dY[b * out_size + j];
                    } 
                    else 
                    {
                        tile_dY[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                    
                    // load tile of W^T (so we access W[i, j] as W_transposed[j, i])
                    j = t * TILE_SIZE + threadIdx.y;
                    if (j < out_size && i < in_size) 
                    {
                        tile_W[threadIdx.y][threadIdx.x] = W[i * out_size + j];
                    } 
                    else 
                    {
                        tile_W[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                    
                    __syncthreads();
                    
                    #pragma unroll
                    for (int k = 0; k < TILE_SIZE; k++) 
                    {
                        sum += tile_dY[threadIdx.y][k] * tile_W[k][threadIdx.x];
                    }
                    
                    __syncthreads();
                }
                
                if (b < batch && i < in_size) 
                {
                    atomicAdd(&dX[b * in_size + i], sum);
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

                matmul_grad_input_kernel<<<grid, block>>>(ddY, dW, ddX, batch, in_size, out_size);

                cudaMemcpy(dX, ddX, batch * in_size * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(ddY);
                cudaFree(ddX);
                cudaFree(dW);
            }
        } 
    } 
}
