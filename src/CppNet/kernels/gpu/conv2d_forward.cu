#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            /**
             * @brief Conv2D forward kernel — one thread per output element.
             *
             * Eigen ColMajor 4D layout: tensor(i0, i1, i2, i3) is at
             *   i0 + d0*i1 + d0*d1*i2 + d0*d1*d2*i3
             */
            __global__ void conv2d_forward_kernel(
                const float* __restrict__ input,    // [N, C_in, H, W]
                const float* __restrict__ weights,  // [C_out, C_in, kH, kW]
                const float* __restrict__ bias,     // [C_out] or nullptr
                float* __restrict__ output,         // [N, C_out, H_out, W_out]
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride, int padding,
                int H_out, int W_out,
                bool use_bias)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total = N * C_out * H_out * W_out;
                if (idx >= total) return;

                // Decode linear index → (n, oc, oh, ow) in ColMajor order
                int n  = idx % N;
                int rem = idx / N;
                int oc = rem % C_out;
                rem = rem / C_out;
                int oh = rem % H_out;
                int ow = rem / H_out;

                float val = 0.0f;
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
                                // input(n, ic, ih, iw) ColMajor
                                int in_idx = n + N * (ic + C_in * (ih + H * iw));
                                // weights(oc, ic, kh, kw_i) ColMajor
                                int w_idx = oc + C_out * (ic + C_in * (kh + kH * kw_i));
                                val += input[in_idx] * weights[w_idx];
                            }
                        }
                    }
                }
                if (use_bias)
                    val += bias[oc];

                // output(n, oc, oh, ow) ColMajor
                output[n + N * (oc + C_out * (oh + H_out * ow))] = val;
            }
        }
    }
}
