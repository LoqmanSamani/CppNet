/**
 * @file max_pool2d.hpp
 * @brief 2D Max Pooling layer
 */

#ifndef MAX_POOL2D_HPP
#define MAX_POOL2D_HPP

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class MaxPool2D
         * @brief 2D Max Pooling with configurable pool size and stride.
         *
         * Input:  [batch, channels, H, W]
         * Output: [batch, channels, H_out, W_out]
         */
        class MaxPool2D : public Layer
        {
        public:
            /**
             * @param pool_size Size of the pooling window (square)
             * @param stride Stride of the pooling (default = pool_size)
             * @param device Compute backend: "cpu-eigen", "cpu", or "gpu"
             */
            MaxPool2D(int pool_size = 2, int stride = -1, const std::string& device = "cpu-eigen");

            Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& input);
            Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output);

            bool is_trainable() const override { return false; }
            void step(Optimizers::Optimizer& /*optimizer*/, float /*learning_rate*/) override {}

            int get_pool_size() const { return pool_size_; }
            int get_stride() const { return stride_; }

        private:
            int pool_size_;
            int stride_;
            std::string device_;
            Eigen::Tensor<int, 4> max_indices_;  // stores argmax for backward
            Eigen::Tensor<float, 4> input_cache_;
        };
    }
}

#endif // MAX_POOL2D_HPP
