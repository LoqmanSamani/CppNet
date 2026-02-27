/**
 * @file rnn_cell.cu
 * @brief CUDA kernels for RNN cell elementwise operations
 *
 * Forward:  h_t = tanh(pre_ih + pre_hh + bias)
 * Backward: dtanh = (grad + dh_next) * (1 - h_t^2)
 *
 * The heavy matmul operations (x_t*W_ih, h_prev*W_hh) are handled by
 * the existing matmul_kernel; these kernels only fuse the cheap
 * element-wise activation / gradient work.
 */

#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
namespace Kernels
{
namespace GPU
{

__global__ void rnn_tanh_forward_kernel(
    const float* pre_ih,   // [batch * hidden]  x_t * W_ih
    const float* pre_hh,   // [batch * hidden]  h_prev * W_hh
    const float* bias,     // [hidden]
    float*       h_t,      // [batch * hidden]  output
    int total,             // batch * hidden
    int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int batch_count = total / hidden;
    int d = idx / batch_count;  // which hidden dimension
    h_t[idx] = tanhf(pre_ih[idx] + pre_hh[idx] + bias[d]);
}

__global__ void rnn_tanh_backward_kernel(
    const float* grad,     // [batch * hidden]  grad_output at this timestep
    const float* dh_next,  // [batch * hidden]  gradient from next timestep
    const float* h_t,      // [batch * hidden]  cached hidden state
    float*       dtanh,    // [batch * hidden]  output pre-activation gradient
    int total,
    int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float dh  = grad[idx] + dh_next[idx];
    float ht  = h_t[idx];
    dtanh[idx] = dh * (1.0f - ht * ht);
}

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
