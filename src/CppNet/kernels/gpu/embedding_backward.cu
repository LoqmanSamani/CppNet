/**
 * @file embedding_backward.cu
 * @brief CUDA kernel for embedding backward pass (scatter-add gradients)
 *
 * Backward: grad_weight[input[n,s], d] += grad_output[n, s, d]
 * Uses atomicAdd for thread-safe scatter accumulation.
 */

#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
namespace Kernels
{
namespace GPU
{

__global__ void embedding_backward_kernel(
    const int*   input,        // [batch * seq_len]  token IDs (row-major)
    const float* grad_output,  // [batch * seq_len * embed_dim] (row-major)
    float*       grad_weight,  // [vocab_size * embed_dim] (row-major)
    int batch_seq,             // batch * seq_len
    int embed_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_seq * embed_dim;
    if (idx >= total) return;

    int pos = idx / embed_dim;   // which (batch, seq) position
    int d   = idx % embed_dim;   // which embedding dimension

    int token_id = input[pos];
    atomicAdd(&grad_weight[token_id * embed_dim + d], grad_output[idx]);
}

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
