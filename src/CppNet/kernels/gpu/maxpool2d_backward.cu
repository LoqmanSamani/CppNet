#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Kernels
    {
        namespace GPU
        {
            /**
             * @brief MaxPool2D backward kernel — routes gradient to max position.
             *
             * Each thread handles one element of grad_output and writes its
             * gradient to the corresponding max position in grad_input via atomicAdd.
             */
            __global__ void maxpool2d_backward_kernel(
                const float* __restrict__ grad_output,  // [N, C, H_out, W_out]
                const int* __restrict__ max_indices,    // [N, C, H_out, W_out]
                float* __restrict__ grad_input,         // [N, C, H, W]
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

                int out_idx = n + N * (ch + C * (oh + H_out * ow));
                float g = grad_output[out_idx];
                int mi = max_indices[out_idx];
                int ph = mi / pool_size;
                int pw = mi % pool_size;
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;

                // grad_input(n, ch, ih, iw)
                atomicAdd(&grad_input[n + N * (ch + C * (ih + H * iw))], g);
            }
        }
    }
}
