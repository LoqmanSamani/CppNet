#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp" 


namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            /**
             * @brief Element-wise operations kernel
             * @param A First input tensor (flattened)
             * @param B Second input tensor (flattened)
             * @param C Output tensor (flattened)
             * @param N Total number of elements
             * @param op Operation: 0=add, 1=subtract, 2=multiply, 3=divide
             */
            __global__ void elementwise_kernel(float* A, float* B, float* C, int N, int op, int /*unused*/)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < N)
                {
                    switch (op)
                    {
                        case 0: C[idx] = A[idx] + B[idx]; break; // add
                        case 1: C[idx] = A[idx] - B[idx]; break; // subtract
                        case 2: C[idx] = A[idx] * B[idx]; break; // multiply
                        case 3: // divide (safe)
                            C[idx] = (B[idx] != 0.0f) ? A[idx] / B[idx] : 0.0f;
                            break;
                        default: C[idx] = A[idx] + B[idx]; break;
                    }
                }
            }


            void elementwise_gpu(const float* A, const float* B, float* C, int N, int op)
            {
                const int block_size = 256;
                int grid_size = (N + block_size - 1) / block_size;

                float *d_A, *d_B, *d_C;
                cudaMalloc(&d_A, N * sizeof(float));
                cudaMalloc(&d_B, N * sizeof(float));
                cudaMalloc(&d_C, N * sizeof(float));

                cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

                elementwise_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, N, op, 0);
                cudaDeviceSynchronize();

                cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
            }

        }
    }
}
