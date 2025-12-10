#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp" 


namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            __global__void elementwise_kernel(float* A, float* B, float* C, int M, int N, int K)
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
                // implementation of memory allocation + kernel launch
            }

        }
    }
}
