#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            __global__ void bias_grad_kernel(
                const float* dY,  // [batch x out_size]
                float* db,        // [out_size]
                int batch, int out_size)
            {
                // shared memory for parallel reduction
                __shared__ float shared_sum[256];  // max 256 threads per block
                
                int j = blockIdx.x; 
                int tid = threadIdx.x;
                
                if (j >= out_size) return;
                
                float thread_sum = 0.0f;
                for (int b = tid; b < batch; b += blockDim.x) 
                {
                    thread_sum += dY[b + j * batch];  // ColMajor
                }
                shared_sum[tid] = thread_sum;
                __syncthreads();
                
                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
                {
                    if (tid < stride) 
                    {
                        shared_sum[tid] += shared_sum[tid + stride];
                    }
                    __syncthreads();
                }
                
                if (tid == 0) 
                {
                    atomicAdd(&db[j], shared_sum[0]);
                }
            }

            void bias_grad_gpu(const float* dY, float* db, int batch, int out)
            {
                float *ddY, *ddb;

                cudaMalloc(&ddY, batch * out * sizeof(float));
                cudaMalloc(&ddb, out * sizeof(float));

                cudaMemcpy(ddY, dY, batch * out * sizeof(float), cudaMemcpyHostToDevice);

                dim3 block(256);
                dim3 grid((out + 255) / 256);

                bias_grad_kernel<<<grid, block>>>(ddY, ddb, batch, out);

                cudaMemcpy(db, ddb, out * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(ddY);
                cudaFree(ddb);
            }
        }
    } 
}
