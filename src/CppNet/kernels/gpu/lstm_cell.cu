/**
 * @file lstm_cell.cu
 * @brief CUDA kernels for LSTM cell elementwise operations
 *
 * Forward:  given pre-concatenated gates (x_t*W_ih + h_prev*W_hh),
 *           apply bias, gate activations, and compute c_t / h_t.
 * Backward: compute dgates [batch, 4*hidden] and dc_out from
 *           dh, dc_next, and cached gate values.
 */

#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
namespace Kernels
{
namespace GPU
{

__global__ void lstm_gates_forward_kernel(
    const float* pre_ih,   // [batch * 4H]  x_t * W_ih
    const float* pre_hh,   // [batch * 4H]  h_prev * W_hh
    const float* bias,     // [4H]
    const float* c_prev,   // [batch * H]
    float*       i_gate,   // [batch * H]
    float*       f_gate,   // [batch * H]
    float*       g_gate,   // [batch * H]
    float*       o_gate,   // [batch * H]
    float*       c_t,      // [batch * H]
    float*       h_t,      // [batch * H]
    float*       tanh_c,   // [batch * H]
    int batch, int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx >= total) return;

    // ColMajor [batch, H]: column d contains batch elements
    int n = idx % batch;   // row (sample index)
    int d = idx / batch;   // column (hidden index)

    // In ColMajor [batch, 4H], element (n, d+k*H) is at n + (d + k*H)*batch
    int base_ih_n = n;  // row offset
    float ig_pre = pre_ih[base_ih_n + (d + 0 * hidden) * batch]
                 + pre_hh[base_ih_n + (d + 0 * hidden) * batch]
                 + bias[d + 0 * hidden];
    float fg_pre = pre_ih[base_ih_n + (d + 1 * hidden) * batch]
                 + pre_hh[base_ih_n + (d + 1 * hidden) * batch]
                 + bias[d + 1 * hidden];
    float gg_pre = pre_ih[base_ih_n + (d + 2 * hidden) * batch]
                 + pre_hh[base_ih_n + (d + 2 * hidden) * batch]
                 + bias[d + 2 * hidden];
    float og_pre = pre_ih[base_ih_n + (d + 3 * hidden) * batch]
                 + pre_hh[base_ih_n + (d + 3 * hidden) * batch]
                 + bias[d + 3 * hidden];

    float ig = 1.0f / (1.0f + expf(-ig_pre));
    float fg = 1.0f / (1.0f + expf(-fg_pre));
    float gg = tanhf(gg_pre);
    float og = 1.0f / (1.0f + expf(-og_pre));

    float ct  = fg * c_prev[idx] + ig * gg;
    float tc  = tanhf(ct);
    float ht  = og * tc;

    i_gate[idx] = ig;
    f_gate[idx] = fg;
    g_gate[idx] = gg;
    o_gate[idx] = og;
    c_t[idx]    = ct;
    tanh_c[idx] = tc;
    h_t[idx]    = ht;
}

__global__ void lstm_gates_backward_kernel(
    const float* dh,       // [batch * H]
    const float* dc_next,  // [batch * H]
    const float* i_gate,
    const float* f_gate,
    const float* g_gate,
    const float* o_gate,
    const float* tanh_c,
    const float* c_prev,   // [batch * H]
    float*       dgates,   // [batch * 4H]  output
    float*       dc_out,   // [batch * H]   dc to pass to prev timestep
    int batch, int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx >= total) return;

    int n = idx % batch;
    int d = idx / batch;

    float tc = tanh_c[idx];
    float dc = dh[idx] * o_gate[idx] * (1.0f - tc * tc) + dc_next[idx];

    float ig = i_gate[idx];
    float fg = f_gate[idx];
    float gg = g_gate[idx];
    float og = o_gate[idx];

    float di = dc * gg       * ig * (1.0f - ig);
    float df = dc * c_prev[idx] * fg * (1.0f - fg);
    float dg = dc * ig       * (1.0f - gg * gg);
    float do_= dh[idx] * tc  * og * (1.0f - og);

    // ColMajor [batch, 4H]: element (n, d+k*H) at n + (d+k*H)*batch
    dgates[n + (d + 0 * hidden) * batch] = di;
    dgates[n + (d + 1 * hidden) * batch] = df;
    dgates[n + (d + 2 * hidden) * batch] = dg;
    dgates[n + (d + 3 * hidden) * batch] = do_;

    dc_out[idx] = dc * fg;
}

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
