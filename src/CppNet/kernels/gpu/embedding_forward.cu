/**
 * @file embedding_forward.cu
 * @brief CUDA kernel for embedding forward pass (gather rows by index)
 *
 * Forward: output[b, s, d] = weight[input[b, s], d]
 * Uses ColMajor layout (Eigen default).
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
    const int*   input,    // [batch, seq_len] ColMajor
    const float* weight,   // [vocab_size, embed_dim] ColMajor
    float*       output,   // [batch, seq_len, embed_dim] ColMajor
    int batch,
    int seq_len,
    int embed_dim,
    int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * embed_dim;
    if (idx >= total) return;

    // Decompose linear index for 3D ColMajor [batch, seq_len, embed_dim]
    // element (b, s, d) at index b + s*batch + d*batch*seq_len
    int bs = batch * seq_len;
    int d = idx / bs;                // embed dimension
    int rem = idx % bs;
    int s = rem / batch;             // seq position
    int b = rem % batch;             // batch index

    // input [batch, seq_len] ColMajor: input(b, s) = input[b + s * batch]
    int token_id = input[b + s * batch];

    // weight [vocab_size, embed_dim] ColMajor: weight(v, d) = weight[v + d * vocab_size]
    float val = weight[token_id + d * vocab_size];

    // output [batch, seq_len, embed_dim] ColMajor: output(b, s, d) = output[b + s*batch + d*batch*seq_len]
    output[b + s * batch + d * bs] = val;
}

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
