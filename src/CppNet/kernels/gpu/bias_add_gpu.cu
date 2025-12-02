#include <cuda_runtime.h>
#include "kernels/gpu.hpp"

__global__
void add_bias_kernel(float* output, const float* bias, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        output[row * N + col] += bias[col];
    }
}

void add_bias_gpu(float* output, const float* bias, int M, int N)
{
    float *dOut, *dBias;

    cudaMalloc(&dOut, M*N*sizeof(float));
    cudaMalloc(&dBias, N*sizeof(float));

    cudaMemcpy(dOut, output, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dBias, bias, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((N+15)/16, (M+15)/16);

    add_bias_kernel<<<grid, block>>>(dOut, dBias, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, dOut, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dOut);
    cudaFree(dBias);
}