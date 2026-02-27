/**
 * @file attention_scores.cu
 * @brief CUDA kernels for attention score computation and softmax
 *
 * 1. attention_scale_kernel: scores *= scale
 * 2. attention_softmax_forward_kernel: row-wise softmax of scores matrix
 * 3. attention_softmax_backward_kernel: Jacobian-vector product for softmax backward
 */

#include <cuda_runtime.h>
#include <cfloat>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
namespace Kernels
{
namespace GPU
{

// scale kernel: scores[i] *= scale
__global__ void attention_scale_kernel(
    float* scores,
    float  scale,
    int    total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    scores[idx] *= scale;
}

// softmax forward: row-wise softmax over [rows × cols] matrix
// each block handles one row (one query position)
__global__ void attention_softmax_forward_kernel(
    const float* scores,   // [rows * cols]
    float*       output,   // [rows * cols]
    int rows,
    int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* sdata = shared;  // shared memory for reductions

    const float* row_in  = scores + row * cols;
    float*       row_out = output + row * cols;

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_max = fmaxf(local_max, row_in[j]);

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        float val = expf(row_in[j] - row_max);
        row_out[j] = val;
        local_sum += val;
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        row_out[j] *= inv_sum;
}

// softmax backward: d_scores[i,j] = attn[i,j] * (grad[i,j] - dot_i)
// each block handles one row
__global__ void attention_softmax_backward_kernel(
    const float* grad_attn,  // [rows * cols]  upstream gradient
    const float* attn,       // [rows * cols]  softmax output (cached)
    float*       grad_scores, // [rows * cols]  output
    int rows,
    int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];

    const float* g_row = grad_attn  + row * cols;
    const float* a_row = attn       + row * cols;
    float*       o_row = grad_scores + row * cols;

    float local_dot = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_dot += a_row[j] * g_row[j];

    shared[threadIdx.x] = local_dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float dot = shared[0];
    __syncthreads();

    // grad_scores[i,j] = attn[i,j] * (grad_attn[i,j] - dot)
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        o_row[j] = a_row[j] * (g_row[j] - dot);
}

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
