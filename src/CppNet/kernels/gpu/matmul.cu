#include <cuda_runtime.h>
#include "CppNet/kernels/gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            __global__void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K)
            {
                // threadIdx: thread index inside a block
                // blockIdx: block index inside the grid
                // blockDim: size of each block
                int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
                int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1

                if (row < M && col < N) 
                {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++)
                        sum += A[row*K + k] * B[k*N + col];
                    C[row*N + col] = sum;
                }
            }


            void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K)
            {
                float *dA, *dB, *dC;

                // allocate gpu memory
                cudaMalloc(&dA, M*K * sizeof(float));
                cudaMalloc(&dB, K*N * sizeof(float));
                cudaMalloc(&dC, M*N * sizeof(float));

                // copy A and B from cpu to gpu
                cudaMemcpy(dA, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(dB, B, K*N*sizeof(float), cudaMemcpyHostToDevice);

                // configure kernel launch
                dim3 block(16, 16);
                dim3 grid((N + 15) / 16, (M + 15) / 16);

                // launch kernel
                matmul_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);

                // wait for gpu to finish
                cudaDeviceSynchronize();

                // copy result C from gpu to cpu
                cudaMemcpy(C, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);

                // free gpu memory
                cudaFree(dA);
                cudaFree(dB);
                cudaFree(dC);
            }
        } 
    } 
} 