/**
 * @file residual.hpp
 * @brief Residual (skip-connection) wrapper layer
 *
 * Wraps a sub-block of layers and adds the input to the output:
 *   output = block(input) + input
 *
 * If the dimensions differ (e.g. different feature count), an optional
 * projection layer is applied to the shortcut path.
 */

#ifndef RESIDUAL_HPP
#define RESIDUAL_HPP

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"
#endif

#include "CppNet/layers/layer.hpp"
#include "CppNet/layers/linear.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>
#include <string>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class Residual
         * @brief Skip-connection wrapper: output = F(x) + shortcut(x)
         *
         * 2D input: [batch, features]
         *
         * If input and output feature dimensions match, shortcut is identity.
         * Otherwise a Linear projection is created automatically.
         */
        class Residual : public Layer
        {
        public:
            /**
             * @param block       Ordered list of layers forming the residual block
             * @param in_features Input feature dimension
             * @param out_features Output feature dimension of last layer in block
             *                     (if != in_features, a projection shortcut is created)
             * @param device      Compute backend
             */
            Residual(std::vector<std::shared_ptr<Layer>> block,
                     int in_features, int out_features,
                     const std::string& device = "cpu-eigen");
            ~Residual() override;

            /// Forward: run the block, then add the (projected) input
            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input);

            /// Backward: gradient flows through both the block and the shortcut
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output);

            bool is_trainable() const override { return true; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

            /// Zero all internal layer gradients
            void reset_grads() override;

            std::size_t num_block_layers() const { return block_.size(); }
            bool has_projection() const { return projection_ != nullptr; }

        private:
            std::vector<std::shared_ptr<Layer>> block_;
            std::shared_ptr<Linear> projection_;  // nullptr when dims match
            int in_features_;
            int out_features_;
            std::string device_;

            /// Cached input for backward
            Eigen::Tensor<float, 2> input_cache_;
        };
    }
}

#endif // RESIDUAL_HPP
