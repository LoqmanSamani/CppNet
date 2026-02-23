/**
 * @file global_pool.cpp
 * @brief GlobalAvgPool2D and GlobalMaxPool2D implementations
 *
 * Input:  [batch, channels, H, W]
 * Output: [batch, channels]
 */

#include "CppNet/layers/global_pool.hpp"
#include <limits>

namespace CppNet
{
    namespace Layers
    {
        const Eigen::Tensor<float, 2> GlobalAvgPool2D::forward(
            const Eigen::Tensor<float, 4>& input)
        {
            batch_cache_    = input.dimension(0);
            channels_cache_ = input.dimension(1);
            height_cache_   = input.dimension(2);
            width_cache_    = input.dimension(3);

            int spatial = height_cache_ * width_cache_;
            float inv_spatial = 1.0f / static_cast<float>(spatial);

            Eigen::Tensor<float, 2> output(batch_cache_, channels_cache_);

            for (int n = 0; n < batch_cache_; ++n)
            {
                for (int c = 0; c < channels_cache_; ++c)
                {
                    float sum = 0.0f;
                    for (int h = 0; h < height_cache_; ++h)
                        for (int w = 0; w < width_cache_; ++w)
                            sum += input(n, c, h, w);
                    output(n, c) = sum * inv_spatial;
                }
            }

            return output;
        }

        const Eigen::Tensor<float, 4> GlobalAvgPool2D::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            int spatial = height_cache_ * width_cache_;
            float inv_spatial = 1.0f / static_cast<float>(spatial);

            Eigen::Tensor<float, 4> grad_input(batch_cache_, channels_cache_,
                                                 height_cache_, width_cache_);

            for (int n = 0; n < batch_cache_; ++n)
                for (int c = 0; c < channels_cache_; ++c)
                {
                    float g = grad_output(n, c) * inv_spatial;
                    for (int h = 0; h < height_cache_; ++h)
                        for (int w = 0; w < width_cache_; ++w)
                            grad_input(n, c, h, w) = g;
                }

            return grad_input;
        }

        void GlobalAvgPool2D::step(Optimizers::Optimizer&, float) {}

        const Eigen::Tensor<float, 2> GlobalMaxPool2D::forward(
            const Eigen::Tensor<float, 4>& input)
        {
            batch_cache_    = input.dimension(0);
            channels_cache_ = input.dimension(1);
            height_cache_   = input.dimension(2);
            width_cache_    = input.dimension(3);

            argmax_h_.resize(batch_cache_, channels_cache_);
            argmax_w_.resize(batch_cache_, channels_cache_);

            Eigen::Tensor<float, 2> output(batch_cache_, channels_cache_);

            for (int n = 0; n < batch_cache_; ++n)
            {
                for (int c = 0; c < channels_cache_; ++c)
                {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int best_h = 0, best_w = 0;

                    for (int h = 0; h < height_cache_; ++h)
                    {
                        for (int w = 0; w < width_cache_; ++w)
                        {
                            if (input(n, c, h, w) > max_val)
                            {
                                max_val = input(n, c, h, w);
                                best_h = h;
                                best_w = w;
                            }
                        }
                    }

                    output(n, c) = max_val;
                    argmax_h_(n, c) = best_h;
                    argmax_w_(n, c) = best_w;
                }
            }

            return output;
        }

        const Eigen::Tensor<float, 4> GlobalMaxPool2D::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            Eigen::Tensor<float, 4> grad_input(batch_cache_, channels_cache_,
                                                 height_cache_, width_cache_);
            grad_input.setZero();

            for (int n = 0; n < batch_cache_; ++n)
                for (int c = 0; c < channels_cache_; ++c)
                    grad_input(n, c, argmax_h_(n, c), argmax_w_(n, c)) =
                        grad_output(n, c);

            return grad_input;
        }

        void GlobalMaxPool2D::step(Optimizers::Optimizer&, float) {}
    }
}
