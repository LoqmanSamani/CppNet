#include <iostream>
#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"




namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            #define TILE_SIZE 32  // 32x32 tiles 

            __global__ void matmul_kernel(
                const float* A,  // input matrix [M x K]  (ColMajor)
                const float* B,  // weight matrix [K x N] (ColMajor)
                float* C,        // output matrix [M x N] (ColMajor)
                int M, int N, int K)
            {
                // each thread computes one element of C
                int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // global row index
                int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // global col index
                
                // shared memory: fast on-chip memory shared by all threads in a block
                __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
                __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
                
                float sum = 0.0f;
                
                // loop over tiles: we break K dimension into TILE_SIZE chunks
                // each iteration loads one tile from A and B, multiplies them
                int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
                
                for (int tile = 0; tile < num_tiles; tile++) 
                {
                    // load one tile from A into shared memory (ColMajor: A(r,c) = A[r + c*M])
                    int a_col = tile * TILE_SIZE + threadIdx.x;
                    if (row < M && a_col < K)
                        tile_A[threadIdx.y][threadIdx.x] = A[row + a_col * M];
                    else
                        tile_A[threadIdx.y][threadIdx.x] = 0.0f;  // padding for edge cases
                    
                    // load one tile from B into shared memory (ColMajor: B(r,c) = B[r + c*K])
                    int b_row = tile * TILE_SIZE + threadIdx.y;
                    if (b_row < K && col < N)
                        tile_B[threadIdx.y][threadIdx.x] = B[b_row + col * K];
                    else
                        tile_B[threadIdx.y][threadIdx.x] = 0.0f;
                    
                    // wait for ALL threads in block to finish loading
                    __syncthreads();
                    
                    // compute partial dot product using the loaded tiles
                    #pragma unroll
                    for (int k = 0; k < TILE_SIZE; k++) 
                    {
                        sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
                    }

                    // wait before loading next tile (prevent data race)
                    __syncthreads();
                }
                
                // write final result to global memory (ColMajor: C(r,c) = C[r + c*M])
                if (row < M && col < N) 
                {
                    C[row + col * M] = sum;
                }
            }


            void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K)
            {
                float *dA, *dB, *dC;
                
                // allocate GPU memory
                cudaMalloc(&dA, M * K * sizeof(float));
                cudaMalloc(&dB, K * N * sizeof(float));
                cudaMalloc(&dC, M * N * sizeof(float));
                
                // copy input data to GPU
                cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
                
                dim3 block(TILE_SIZE, TILE_SIZE);
                dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
                
                // launch kernel
                matmul_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
                
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) 
                {
                    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
                }
                
                // wait for GPU to finish
                cudaDeviceSynchronize();
                
                // copy result back to CPU
                cudaMemcpy(C, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
                
                // free GPU memory
                cudaFree(dA);
                cudaFree(dB);
                cudaFree(dC);
            }
        } 
    } 
} 