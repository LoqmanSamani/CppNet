/**
 * @file mean_pool1d.cu
 * @brief CUDA kernels for MeanPool1D forward and backward
 *
 * Forward:  output[b,d] = (1/S) * sum_s input[b,s,d]
 * Backward: grad_input[b,s,d] = grad_output[b,d] / S
 *
 * All arrays use ColMajor (Eigen default).
 *   3D [B,S,D]:  element (b,s,d)  at  b + s*B + d*B*S
 *   2D [B,D]:    element (b,d)    at  b + d*B
 */

#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
namespace Kernels
{
namespace GPU
{

/**
 * @brief Forward kernel — one thread per (b, d),
 *        sums over S sequence positions and divides by S.
 */
__global__ void mean_pool1d_forward_kernel(
    const float* input,   // [B, S, D] ColMajor
    float*       output,  // [B, D]    ColMajor
    int B, int S, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D;
    if (idx >= total) return;

    // decompose 2D ColMajor [B, D]: idx = b + d*B
    int d = idx / B;
    int b = idx % B;

    float sum = 0.0f;
    for (int s = 0; s < S; ++s)
        sum += input[b + s * B + d * B * S];   // input(b,s,d)

    output[b + d * B] = sum / static_cast<float>(S);   // output(b,d)
}

/**
 * @brief Backward kernel — one thread per (b, s, d),
 *        broadcasts grad_output and divides by S.
 */
__global__ void mean_pool1d_backward_kernel(
    const float* grad_output,  // [B, D]    ColMajor
    float*       grad_input,   // [B, S, D] ColMajor
    int B, int S, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (idx >= total) return;

    // decompose 3D ColMajor [B, S, D]: idx = b + s*B + d*B*S
    int BS = B * S;
    int d  = idx / BS;
    int rem = idx % BS;
    int s  = rem / B;
    int b  = rem % B;

    float inv_S = 1.0f / static_cast<float>(S);
    grad_input[idx] = grad_output[b + d * B] * inv_S;   // grad_output(b,d) / S
}

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
