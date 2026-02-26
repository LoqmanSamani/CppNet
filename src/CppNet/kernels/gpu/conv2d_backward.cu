#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            /**
             * @brief Conv2D backward kernel — computes grad_input, grad_weights, grad_biases.
             *
             * Each thread handles one element of grad_output and scatters contributions
             * to grad_input and grad_weights via atomicAdd.
             *
             * Eigen ColMajor 4D layout: tensor(i0, i1, i2, i3) is at
             *   i0 + d0*i1 + d0*d1*i2 + d0*d1*d2*i3
             */
            __global__ void conv2d_backward_kernel(
                const float* __restrict__ grad_output,  // [N, C_out, H_out, W_out]
                const float* __restrict__ input_cache,   // [N, C_in, H, W]
                const float* __restrict__ weights,       // [C_out, C_in, kH, kW]
                float* __restrict__ grad_input,          // [N, C_in, H, W]
                float* __restrict__ grad_weights,        // [C_out, C_in, kH, kW]
                float* __restrict__ grad_biases,         // [C_out] or nullptr
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride, int padding,
                int H_out, int W_out,
                bool use_bias)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total = N * C_out * H_out * W_out;
                if (idx >= total) return;

                // decode linear index → (n, oc, oh, ow) in ColMajor order
                int n  = idx % N;
                int rem = idx / N;
                int oc = rem % C_out;
                rem = rem / C_out;
                int oh = rem % H_out;
                int ow = rem / H_out;

                float g = grad_output[n + N * (oc + C_out * (oh + H_out * ow))];

                if (use_bias && grad_biases)
                    atomicAdd(&grad_biases[oc], g);

                for (int ic = 0; ic < C_in; ++ic)
                {
                    for (int kh = 0; kh < kH; ++kh)
                    {
                        for (int kw_i = 0; kw_i < kW; ++kw_i)
                        {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw_i;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            {
                                int in_idx = n + N * (ic + C_in * (ih + H * iw));
                                int w_idx = oc + C_out * (ic + C_in * (kh + kH * kw_i));

                                atomicAdd(&grad_weights[w_idx], input_cache[in_idx] * g);
                                atomicAdd(&grad_input[in_idx], weights[w_idx] * g);
                            }
                        }
                    }
                }
            }
        }
    }
}
