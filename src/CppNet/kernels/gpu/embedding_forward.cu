/**
 * @file embedding_forward.cu
 * @brief CUDA kernel for embedding forward pass (gather rows by index)
 *
 * Forward: output[n, s, d] = weight[input[n, s], d]
 * Each thread handles one element of the output tensor.
 */

#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
namespace Kernels
{
namespace GPU
{

__global__ void embedding_forward_kernel(
    const int*   input,    // [batch * seq_len]  token IDs (row-major)
    const float* weight,   // [vocab_size * embed_dim] (row-major)
    float*       output,   // [batch * seq_len * embed_dim] (row-major)
    int batch_seq,         // batch * seq_len
    int embed_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_seq * embed_dim;
    if (idx >= total) return;

    int pos = idx / embed_dim;   // which (batch, seq) position
    int d   = idx % embed_dim;   // which embedding dimension

    int token_id = input[pos];
    output[idx] = weight[token_id * embed_dim + d];
}

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
