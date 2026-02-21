/**
 * @file global_pool.hpp
 * @brief Global Average Pooling and Global Max Pooling layers
 *
 * Reduces a 4D tensor [batch, channels, H, W] to [batch, channels]
 * by averaging or taking the max over the spatial dimensions.
 * Commonly used before the final classifier head in CNNs.
 */

#ifndef GLOBAL_POOL_HPP
#define GLOBAL_POOL_HPP

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class GlobalAvgPool2D
         * @brief Global average pooling over spatial dims
         *
         * Input:  [batch, channels, H, W]
         * Output: [batch, channels]
         */
        class GlobalAvgPool2D : public Layer
        {
        public:
            GlobalAvgPool2D() = default;
            ~GlobalAvgPool2D() override = default;

            const Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 4>& input);
            const Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 2>& grad_output);

            bool is_trainable() const override { return false; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

        private:
            int batch_cache_ = 0;
            int channels_cache_ = 0;
            int height_cache_ = 0;
            int width_cache_ = 0;
        };

        /**
         * @class GlobalMaxPool2D
         * @brief Global max pooling over spatial dims
         *
         * Input:  [batch, channels, H, W]
         * Output: [batch, channels]
         */
        class GlobalMaxPool2D : public Layer
        {
        public:
            GlobalMaxPool2D() = default;
            ~GlobalMaxPool2D() override = default;

            const Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 4>& input);
            const Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 2>& grad_output);

            bool is_trainable() const override { return false; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

        private:
            int batch_cache_ = 0;
            int channels_cache_ = 0;
            int height_cache_ = 0;
            int width_cache_ = 0;

            /// Stores argmax (h, w) per (batch, channel) for gradient routing
            Eigen::Tensor<int, 2> argmax_h_;  // [batch, channels]
            Eigen::Tensor<int, 2> argmax_w_;  // [batch, channels]
        };
    }
}

#endif // GLOBAL_POOL_HPP
