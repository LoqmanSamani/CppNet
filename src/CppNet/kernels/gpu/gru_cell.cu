/**
 * @file gru_cell.cu
 * @brief CUDA kernels for GRU cell elementwise operations
 *
 * Five small kernels that handle the non-matmul parts of the GRU
 * forward and backward passes.  The matmuls (x_t*W_ih, h_prev*W_hh,
 * rh*W_hn, etc.) are handled by the existing matmul kernels.
 */

#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
namespace Kernels
{
namespace GPU
{

__global__ void gru_zr_forward_kernel(
    const float* x_gates,  // [batch * 3H]  x_t * W_ih   (ColMajor)
    const float* h_gates,  // [batch * 3H]  h_prev * W_hh
    const float* bias,     // [3H]
    const float* h_prev,   // [batch * H]
    float*       z_gate,   // [batch * H]
    float*       r_gate,   // [batch * H]
    float*       rh,       // [batch * H]  r ◦ h_prev
    int batch, int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * hidden) return;

    int n = idx % batch;
    int d = idx / batch;

    // ColMajor [batch, 3H]: element (n, d+k*H) at n + (d+k*H)*batch
    float z_pre = x_gates[n + (d + 0 * hidden) * batch]
                + h_gates[n + (d + 0 * hidden) * batch]
                + bias[d + 0 * hidden];
    float r_pre = x_gates[n + (d + 1 * hidden) * batch]
                + h_gates[n + (d + 1 * hidden) * batch]
                + bias[d + 1 * hidden];

    float z = 1.0f / (1.0f + expf(-z_pre));
    float r = 1.0f / (1.0f + expf(-r_pre));

    z_gate[idx] = z;
    r_gate[idx] = r;
    rh[idx]     = r * h_prev[idx];
}

__global__ void gru_output_forward_kernel(
    const float* x_gates,  // [batch * 3H]  (only the n-column used)
    const float* rh_proj,  // [batch * H]   rh * W_hn
    const float* bias,     // [3H]          (only n-bias used)
    const float* z_gate,   // [batch * H]
    const float* h_prev,   // [batch * H]
    float*       n_cand,   // [batch * H]
    float*       h_t,      // [batch * H]
    int batch, int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * hidden) return;

    int n = idx % batch;
    int d = idx / batch;

    float nc = tanhf(x_gates[n + (d + 2 * hidden) * batch]
                   + rh_proj[idx]
                   + bias[d + 2 * hidden]);
    float z  = z_gate[idx];

    n_cand[idx] = nc;
    h_t[idx]    = (1.0f - z) * nc + z * h_prev[idx];
}

__global__ void gru_bwd_gates_kernel(
    const float* dh_raw,     // [batch * H]
    const float* dh_next_in, // [batch * H]
    const float* z_gate,
    const float* n_cand,
    const float* h_prev,
    float*       dn_raw,     // [batch * H]
    float*       dz_raw,     // [batch * H]
    float*       dh_prev_z,  // [batch * H]  partial: dh * z
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float dh = dh_raw[idx] + dh_next_in[idx];
    float z  = z_gate[idx];
    float nc = n_cand[idx];
    float hp = h_prev[idx];

    float dn = dh * (1.0f - z);
    float dz = dh * (hp - nc);

    dn_raw[idx]   = dn * (1.0f - nc * nc);        // tanh derivative
    dz_raw[idx]   = dz * z * (1.0f - z);          // sigmoid derivative
    dh_prev_z[idx] = dh * z;
}

__global__ void gru_bwd_reset_kernel(
    const float* d_rh,       // [batch * H]  grad w.r.t. rh (from dn_raw * W_hn^T)
    const float* h_prev,
    const float* r_gate,
    const float* dh_prev_z,  // [batch * H]  partial from step 1
    float*       dr_raw,     // [batch * H]
    float*       dh_prev_out,// [batch * H]  complete dh_prev
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float drh = d_rh[idx];
    float hp  = h_prev[idx];
    float r   = r_gate[idx];

    float dr = drh * hp;
    dr_raw[idx]      = dr * r * (1.0f - r);
    dh_prev_out[idx] = dh_prev_z[idx] + drh * r;
}

__global__ void gru_assemble_dgates_kernel(
    const float* dz_raw,  // [batch * H]
    const float* dr_raw,  // [batch * H]
    const float* dn_raw,  // [batch * H]
    float*       dgates,  // [batch * 3H]
    int batch, int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx >= total) return;

    // ColMajor: [batch, 3H] gate-block k starts at offset k * H * batch
    dgates[idx]                          = dz_raw[idx];
    dgates[idx + hidden * batch]         = dr_raw[idx];
    dgates[idx + 2 * hidden * batch]     = dn_raw[idx];
}

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
