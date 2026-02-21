/**
 * @file flatten.hpp
 * @brief Flatten layer for reshaping 4D tensors to 2D
 */

#ifndef FLATTEN_HPP
#define FLATTEN_HPP

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <array>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class Flatten
         * @brief Flattens a 4D tensor [batch, C, H, W] into 2D [batch, C*H*W]
         *
         * This layer has no trainable parameters. It simply reshapes
         * the input tensor while preserving the batch dimension.
         */
        class Flatten : public Layer
        {
        public:
            Flatten() = default;

            /**
             * @brief Forward: reshape [batch, C, H, W] -> [batch, C*H*W]
             */
            const Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 4>& input);

            /**
             * @brief Backward: reshape [batch, C*H*W] -> [batch, C, H, W]
             */
            const Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 2>& grad_output);

            bool is_trainable() const override { return false; }
            void step(Optimizers::Optimizer& /*optimizer*/, float /*learning_rate*/) override {}

        private:
            std::array<int, 4> input_shape_; // cached for backward
        };
    }
}

#endif // FLATTEN_HPP
