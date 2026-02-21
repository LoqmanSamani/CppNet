/**
 * @file conv2d.cpp
 * @brief 2D Convolutional layer implementation
 *
 * Implements forward and backward passes for 2D convolution.
 * Supports "cpu-eigen", "cpu" (OpenMP), and "gpu" (CUDA) backends.
 */

#include "CppNet/layers/conv2d.hpp"
#include "CppNet/utils/init.hpp"
#include <stdexcept>
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
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

            // Xavier initialization
            float fan_in = static_cast<float>(in_channels_ * kernel_size_ * kernel_size_);
            float fan_out = static_cast<float>(out_channels_ * kernel_size_ * kernel_size_);
            float limit = std::sqrt(6.0f / (fan_in + fan_out));

            weights_.setRandom();
            weights_ = weights_ * weights_.constant(limit);

            if (bias_flag_)
            {
                biases_.resize(out_channels_);
                biases_.setZero();
                grad_biases_.resize(out_channels_);
                grad_biases_.setZero();
            }
        }

        Conv2D::~Conv2D() = default;

        const Eigen::Tensor<float, 4> Conv2D::forward(const Eigen::Tensor<float, 4>& input)
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

        const Eigen::Tensor<float, 4> Conv2D::backward(const Eigen::Tensor<float, 4>& grad_output)
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

        void Conv2D::step(Optimizers::Optimizer& /*optimizer*/, float /*learning_rate*/)
        {
            // TODO: Integrate with optimizer once it supports Conv2D
            // For now, manual SGD-like update can be done externally
        }
    }
}
