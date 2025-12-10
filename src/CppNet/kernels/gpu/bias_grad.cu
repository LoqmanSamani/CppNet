#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            __global__ void bias_grad_kernel(const float* dY, float* db, int batch, int out)
            {
                int j = blockIdx.x * blockDim.x + threadIdx.x;
                if (j >= out) return;

                float sum = 0.f;
                for (int b = 0; b < batch; b++)
                {
                    sum += dY[b * out + j];
                }
                db[j] = sum;
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
