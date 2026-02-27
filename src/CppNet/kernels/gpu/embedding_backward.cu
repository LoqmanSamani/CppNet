/**
 * @file embedding_backward.cu
 * @brief CUDA kernel for embedding backward pass (scatter-add gradients)
 *
 * Backward: grad_weight[input[b,s], d] += grad_output[b, s, d]
 * Uses ColMajor layout (Eigen default) and atomicAdd for thread safety.
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
    const int*   input,        // [batch, seq_len] ColMajor
    const float* grad_output,  // [batch, seq_len, embed_dim] ColMajor
    float*       grad_weight,  // [vocab_size, embed_dim] ColMajor
    int batch,
    int seq_len,
    int embed_dim,
    int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * embed_dim;
    if (idx >= total) return;

    int bs = batch * seq_len;
    int d = idx / bs;
    int rem = idx % bs;
    int s = rem / batch;
    int b = rem % batch;

    int token_id = input[b + s * batch];

    float grad_val = grad_output[b + s * batch + d * bs];

    atomicAdd(&grad_weight[token_id + d * vocab_size], grad_val);
}

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
