/**
 * @file conv2d.cpp
 * @brief 2D Convolutional layer implementation
 *
 * Implements forward and backward passes for 2D convolution.
 * Supports "cpu-eigen", "cpu" (OpenMP), and "gpu" (CUDA) backends.
 */

#include "CppNet/layers/conv2d.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/utils/init.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>

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
        Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size,
                       int stride, int padding, bool bias,
                       const std::string& device)
            : in_channels_(in_channels), out_channels_(out_channels),
              kernel_size_(kernel_size), stride_(stride), padding_(padding),
              bias_flag_(bias), device_(device)
        {
            // Initialize weights: [out_channels, in_channels, kH, kW]
            weights_.resize(out_channels_, in_channels_, kernel_size_, kernel_size_);
            grad_weights_.resize(out_channels_, in_channels_, kernel_size_, kernel_size_);
            grad_weights_.setZero();

            // Xavier initialization via utility (flatten 4D → 2D, copy back)
            float fan_in = static_cast<float>(in_channels_ * kernel_size_ * kernel_size_);
            float fan_out = static_cast<float>(out_channels_ * kernel_size_ * kernel_size_);
            float limit = std::sqrt(6.0f / (fan_in + fan_out));

            auto w_flat = CppNet::Utils::uniform_init(
                out_channels_, in_channels_ * kernel_size_ * kernel_size_,
                -limit, limit);
            std::memcpy(weights_.data(), w_flat.data(),
                        weights_.size() * sizeof(float));

            if (bias_flag_)
            {
                biases_.resize(out_channels_);
                biases_.setZero();
                grad_biases_.resize(out_channels_);
                grad_biases_.setZero();
            }
        }

        Conv2D::~Conv2D() = default;

        #ifdef USE_CUDA
        void Conv2D::set_max_batch_size(int max_batch_size)
        {
            gpu_max_batch_size_ = max_batch_size;
            std::cout << "Conv2D GPU max batch size set to " << max_batch_size << "\n";
        }
        #endif

        void Conv2D::reset_grads()
        {
            grad_weights_.setZero();
            if (bias_flag_)
                grad_biases_.setZero();
        }

        Eigen::Tensor<float, 4> Conv2D::forward(const Eigen::Tensor<float, 4>& input)
        {
            // input: [batch, in_channels, H, W]
            input_cache_ = input;

            int batch = input.dimension(0);
            int H = input.dimension(2);
            int W = input.dimension(3);

            int H_out = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
            int W_out = (W + 2 * padding_ - kernel_size_) / stride_ + 1;

            Eigen::Tensor<float, 4> output(batch, out_channels_, H_out, W_out);
            output.setZero();

            #ifdef USE_CUDA
            if (device_ == "gpu")
            {
                forward_gpu(input, output, batch, H, W, H_out, W_out);
                return output;
            }
            #endif

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2) if(device_ == "cpu")
            #endif
            for (int n = 0; n < batch; ++n)
            {
                for (int oc = 0; oc < out_channels_; ++oc)
                {
                    for (int oh = 0; oh < H_out; ++oh)
                    {
                        for (int ow = 0; ow < W_out; ++ow)
                        {
                            float val = 0.0f;
                            for (int ic = 0; ic < in_channels_; ++ic)
                            {
                                for (int kh = 0; kh < kernel_size_; ++kh)
                                {
                                    for (int kw = 0; kw < kernel_size_; ++kw)
                                    {
                                        int ih = oh * stride_ - padding_ + kh;
                                        int iw = ow * stride_ - padding_ + kw;

                                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                            val += input(n, ic, ih, iw) * weights_(oc, ic, kh, kw);
                                    }
                                }
                            }
                            if (bias_flag_)
                                val += biases_(oc);
                            output(n, oc, oh, ow) = val;
                        }
                    }
                }
            }

            return output;
        }

        Eigen::Tensor<float, 4> Conv2D::backward(const Eigen::Tensor<float, 4>& grad_output)
        {
            // grad_output: [batch, out_channels, H_out, W_out]
            int batch = grad_output.dimension(0);
            int H_out = grad_output.dimension(2);
            int W_out = grad_output.dimension(3);
            int H = input_cache_.dimension(2);
            int W = input_cache_.dimension(3);

            Eigen::Tensor<float, 4> grad_input(batch, in_channels_, H, W);
            grad_input.setZero();
            grad_weights_.setZero();

            if (bias_flag_)
                grad_biases_.setZero();

            #ifdef USE_CUDA
            if (device_ == "gpu")
            {
                backward_gpu(grad_output, grad_input, batch, H, W, H_out, W_out);
                return grad_input;
            }
            #endif

            for (int n = 0; n < batch; ++n)
            {
                for (int oc = 0; oc < out_channels_; ++oc)
                {
                    for (int oh = 0; oh < H_out; ++oh)
                    {
                        for (int ow = 0; ow < W_out; ++ow)
                        {
                            float g = grad_output(n, oc, oh, ow);

                            if (bias_flag_)
                                grad_biases_(oc) += g;

                            for (int ic = 0; ic < in_channels_; ++ic)
                            {
                                for (int kh = 0; kh < kernel_size_; ++kh)
                                {
                                    for (int kw = 0; kw < kernel_size_; ++kw)
                                    {
                                        int ih = oh * stride_ - padding_ + kh;
                                        int iw = ow * stride_ - padding_ + kw;

                                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                        {
                                            grad_weights_(oc, ic, kh, kw) += input_cache_(n, ic, ih, iw) * g;
                                            grad_input(n, ic, ih, iw) += weights_(oc, ic, kh, kw) * g;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return grad_input;
        }

        void Conv2D::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            if (!trainable_) return;

            optimizer.update(weights_.data(), grad_weights_.data(),
                             weights_.size(), learning_rate);

            if (bias_flag_)
                optimizer.update(biases_.data(), grad_biases_.data(),
                                 biases_.size(), learning_rate);

            grad_weights_.setZero();
            grad_biases_.setZero();
        }

        // ─── GPU forward/backward ─────────────────────────────────────

        void Conv2D::forward_gpu(
            [[maybe_unused]] const Eigen::Tensor<float, 4>& input,
            [[maybe_unused]] Eigen::Tensor<float, 4>& output,
            [[maybe_unused]] int batch, [[maybe_unused]] int H,
            [[maybe_unused]] int W, [[maybe_unused]] int H_out,
            [[maybe_unused]] int W_out)
        {
            #ifdef USE_CUDA
            int in_size  = batch * in_channels_ * H * W;
            int out_size = batch * out_channels_ * H_out * W_out;
            int w_size   = out_channels_ * in_channels_ * kernel_size_ * kernel_size_;

            float *d_input, *d_weights, *d_bias, *d_output;
            cudaMalloc(&d_input,   in_size  * sizeof(float));
            cudaMalloc(&d_weights, w_size   * sizeof(float));
            cudaMalloc(&d_output,  out_size * sizeof(float));

            cudaMemcpy(d_input,   input.data(),   in_size  * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weights, weights_.data(), w_size   * sizeof(float), cudaMemcpyHostToDevice);

            d_bias = nullptr;
            if (bias_flag_)
            {
                cudaMalloc(&d_bias, out_channels_ * sizeof(float));
                cudaMemcpy(d_bias, biases_.data(), out_channels_ * sizeof(float), cudaMemcpyHostToDevice);
            }

            int total = out_size;
            int block = 256;
            int grid  = (total + block - 1) / block;

            Kernels::GPU::conv2d_forward_kernel<<<grid, block>>>(
                d_input, d_weights, d_bias, d_output,
                batch, in_channels_, H, W,
                out_channels_, kernel_size_, kernel_size_,
                stride_, padding_, H_out, W_out, bias_flag_);
            cudaDeviceSynchronize();

            cudaMemcpy(output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_input);
            cudaFree(d_weights);
            cudaFree(d_output);
            if (d_bias) cudaFree(d_bias);
            #endif
        }

        void Conv2D::backward_gpu(
            [[maybe_unused]] const Eigen::Tensor<float, 4>& grad_output,
            [[maybe_unused]] Eigen::Tensor<float, 4>& grad_input,
            [[maybe_unused]] int batch, [[maybe_unused]] int H,
            [[maybe_unused]] int W, [[maybe_unused]] int H_out,
            [[maybe_unused]] int W_out)
        {
            #ifdef USE_CUDA
            int in_size   = batch * in_channels_ * H * W;
            int out_size  = batch * out_channels_ * H_out * W_out;
            int w_size    = out_channels_ * in_channels_ * kernel_size_ * kernel_size_;

            float *d_grad_out, *d_input_cache, *d_weights;
            float *d_grad_input, *d_grad_weights, *d_grad_biases;

            cudaMalloc(&d_grad_out,    out_size * sizeof(float));
            cudaMalloc(&d_input_cache, in_size  * sizeof(float));
            cudaMalloc(&d_weights,     w_size   * sizeof(float));
            cudaMalloc(&d_grad_input,  in_size  * sizeof(float));
            cudaMalloc(&d_grad_weights, w_size  * sizeof(float));

            cudaMemcpy(d_grad_out,    grad_output.data(),   out_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_cache, input_cache_.data(),  in_size  * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weights,     weights_.data(),      w_size   * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_grad_input,  0, in_size  * sizeof(float));
            cudaMemset(d_grad_weights, 0, w_size  * sizeof(float));

            d_grad_biases = nullptr;
            if (bias_flag_)
            {
                cudaMalloc(&d_grad_biases, out_channels_ * sizeof(float));
                cudaMemset(d_grad_biases, 0, out_channels_ * sizeof(float));
            }

            int total = out_size;
            int block = 256;
            int grid  = (total + block - 1) / block;

            Kernels::GPU::conv2d_backward_kernel<<<grid, block>>>(
                d_grad_out, d_input_cache, d_weights,
                d_grad_input, d_grad_weights, d_grad_biases,
                batch, in_channels_, H, W,
                out_channels_, kernel_size_, kernel_size_,
                stride_, padding_, H_out, W_out, bias_flag_);
            cudaDeviceSynchronize();

            cudaMemcpy(grad_input.data(),    d_grad_input,   in_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_weights_.data(), d_grad_weights, w_size  * sizeof(float), cudaMemcpyDeviceToHost);
            if (bias_flag_)
                cudaMemcpy(grad_biases_.data(), d_grad_biases, out_channels_ * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_grad_out);
            cudaFree(d_input_cache);
            cudaFree(d_weights);
            cudaFree(d_grad_input);
            cudaFree(d_grad_weights);
            if (d_grad_biases) cudaFree(d_grad_biases);
            #endif
        }

    }
}
