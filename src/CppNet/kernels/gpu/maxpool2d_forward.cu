#include <cuda_runtime.h>
#include <float.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            /**
             * @brief MaxPool2D forward kernel — one thread per output element.
             *
             * Eigen ColMajor 4D layout: tensor(i0, i1, i2, i3) is at
             *   i0 + d0*i1 + d0*d1*i2 + d0*d1*d2*i3
             */
            __global__ void maxpool2d_forward_kernel(
                const float* __restrict__ input,   // [N, C, H, W]
                float* __restrict__ output,        // [N, C, H_out, W_out]
                int* __restrict__ max_indices,     // [N, C, H_out, W_out]
                int N, int C, int H, int W,
                int pool_size, int stride,
                int H_out, int W_out)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total = N * C * H_out * W_out;
                if (idx >= total) return;

                // decode: (n, c, oh, ow) ColMajor
                int n   = idx % N;
                int rem = idx / N;
                int ch  = rem % C;
                rem = rem / C;
                int oh  = rem % H_out;
                int ow  = rem / H_out;

                float max_val = -FLT_MAX;
                int max_idx = 0;

                for (int ph = 0; ph < pool_size; ++ph)
                {
                    for (int pw = 0; pw < pool_size; ++pw)
                    {
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;
                        // input(n, ch, ih, iw)
                        float val = input[n + N * (ch + C * (ih + H * iw))];
                        if (val > max_val)
                        {
                            max_val = val;
                            max_idx = ph * pool_size + pw;
                        }
                    }
                }

                int out_idx = n + N * (ch + C * (oh + H_out * ow));
                output[out_idx] = max_val;
                max_indices[out_idx] = max_idx;
            }
        }
    }
}
