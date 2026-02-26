/**
 * @file max_pool2d.cpp
 * @brief 2D Max Pooling layer implementation
 *
 * Forward: selects max value in each pooling window.
 * Backward: routes gradient only to the position of the max value.
 */

#include "CppNet/layers/max_pool2d.hpp"
#include <algorithm>
#include <limits>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "CppNet/kernels/gpu/gpu.hpp"
#endif

namespace CppNet
{
    namespace Layers
    {
        MaxPool2D::MaxPool2D(int pool_size, int stride, const std::string& device)
            : pool_size_(pool_size),
              stride_(stride > 0 ? stride : pool_size),
              device_(device)
        {
        }

        Eigen::Tensor<float, 4> MaxPool2D::forward(const Eigen::Tensor<float, 4>& input)
        {
            // input: [batch, channels, H, W]
            input_cache_ = input;

            int batch = input.dimension(0);
            int channels = input.dimension(1);
            int H = input.dimension(2);
            int W = input.dimension(3);

            int H_out = (H - pool_size_) / stride_ + 1;
            int W_out = (W - pool_size_) / stride_ + 1;

            Eigen::Tensor<float, 4> output(batch, channels, H_out, W_out);
            max_indices_.resize(batch, channels, H_out, W_out);

            #ifdef USE_CUDA
            if (device_ == "gpu")
            {
                forward_gpu(input, output, batch, channels, H, W, H_out, W_out);
                return output;
            }
            #endif

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2) if(device_ == "cpu")
            #endif
            for (int n = 0; n < batch; ++n)
            {
                for (int c = 0; c < channels; ++c)
                {
                    for (int oh = 0; oh < H_out; ++oh)
                    {
                        for (int ow = 0; ow < W_out; ++ow)
                        {
                            float max_val = -std::numeric_limits<float>::infinity();
                            int max_idx = 0;

                            for (int ph = 0; ph < pool_size_; ++ph)
                            {
                                for (int pw = 0; pw < pool_size_; ++pw)
                                {
                                    int ih = oh * stride_ + ph;
                                    int iw = ow * stride_ + pw;
                                    float val = input(n, c, ih, iw);

                                    if (val > max_val)
                                    {
                                        max_val = val;
                                        max_idx = ph * pool_size_ + pw;
                                    }
                                }
                            }

                            output(n, c, oh, ow) = max_val;
                            max_indices_(n, c, oh, ow) = max_idx;
                        }
                    }
                }
            }

            return output;
        }

        Eigen::Tensor<float, 4> MaxPool2D::backward(const Eigen::Tensor<float, 4>& grad_output)
        {
            int batch = input_cache_.dimension(0);
            int channels = input_cache_.dimension(1);
            int H = input_cache_.dimension(2);
            int W = input_cache_.dimension(3);
            int H_out = grad_output.dimension(2);
            int W_out = grad_output.dimension(3);

            Eigen::Tensor<float, 4> grad_input(batch, channels, H, W);
            grad_input.setZero();

            #ifdef USE_CUDA
            if (device_ == "gpu")
            {
                backward_gpu(grad_output, grad_input, batch, channels, H, W, H_out, W_out);
                return grad_input;
            }
            #endif

            for (int n = 0; n < batch; ++n)
                for (int c = 0; c < channels; ++c)
                    for (int oh = 0; oh < H_out; ++oh)
                        for (int ow = 0; ow < W_out; ++ow)
                        {
                            int idx = max_indices_(n, c, oh, ow);
                            int ph = idx / pool_size_;
                            int pw = idx % pool_size_;
                            int ih = oh * stride_ + ph;
                            int iw = ow * stride_ + pw;

                            grad_input(n, c, ih, iw) += grad_output(n, c, oh, ow);
                        }

            return grad_input;
        }

        // ─── GPU forward/backward ─────────────────────────────────────

        void MaxPool2D::forward_gpu(
            [[maybe_unused]] const Eigen::Tensor<float, 4>& input,
            [[maybe_unused]] Eigen::Tensor<float, 4>& output,
            [[maybe_unused]] int batch, [[maybe_unused]] int channels,
            [[maybe_unused]] int H, [[maybe_unused]] int W,
            [[maybe_unused]] int H_out, [[maybe_unused]] int W_out)
        {
            #ifdef USE_CUDA
            int in_size  = batch * channels * H * W;
            int out_size = batch * channels * H_out * W_out;

            float *d_input, *d_output;
            int   *d_max_indices;

            cudaMalloc(&d_input,       in_size  * sizeof(float));
            cudaMalloc(&d_output,      out_size * sizeof(float));
            cudaMalloc(&d_max_indices, out_size * sizeof(int));

            cudaMemcpy(d_input, input.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);

            int total = out_size;
            int block = 256;
            int grid  = (total + block - 1) / block;

            Kernels::GPU::maxpool2d_forward_kernel<<<grid, block>>>(
                d_input, d_output, d_max_indices,
                batch, channels, H, W,
                pool_size_, stride_, H_out, W_out);
            cudaDeviceSynchronize();

            cudaMemcpy(output.data(),       d_output,      out_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(max_indices_.data(), d_max_indices, out_size * sizeof(int),   cudaMemcpyDeviceToHost);

            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_max_indices);
            #endif
        }

        void MaxPool2D::backward_gpu(
            [[maybe_unused]] const Eigen::Tensor<float, 4>& grad_output,
            [[maybe_unused]] Eigen::Tensor<float, 4>& grad_input,
            [[maybe_unused]] int batch, [[maybe_unused]] int channels,
            [[maybe_unused]] int H, [[maybe_unused]] int W,
            [[maybe_unused]] int H_out, [[maybe_unused]] int W_out)
        {
            #ifdef USE_CUDA
            int in_size  = batch * channels * H * W;
            int out_size = batch * channels * H_out * W_out;

            float *d_grad_out, *d_grad_in;
            int   *d_max_indices;

            cudaMalloc(&d_grad_out,    out_size * sizeof(float));
            cudaMalloc(&d_grad_in,     in_size  * sizeof(float));
            cudaMalloc(&d_max_indices, out_size * sizeof(int));

            cudaMemcpy(d_grad_out,    grad_output.data(),   out_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_max_indices, max_indices_.data(),  out_size * sizeof(int),   cudaMemcpyHostToDevice);
            cudaMemset(d_grad_in, 0, in_size * sizeof(float));

            int total = out_size;
            int block = 256;
            int grid  = (total + block - 1) / block;

            Kernels::GPU::maxpool2d_backward_kernel<<<grid, block>>>(
                d_grad_out, d_max_indices, d_grad_in,
                batch, channels, H, W,
                pool_size_, stride_, H_out, W_out);
            cudaDeviceSynchronize();

            cudaMemcpy(grad_input.data(), d_grad_in, in_size * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_grad_out);
            cudaFree(d_grad_in);
            cudaFree(d_max_indices);
            #endif
        }

    }
}
