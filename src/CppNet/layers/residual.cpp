/**
 * @file residual.cpp
 * @brief Residual (skip-connection) wrapper implementation
 *
 * Forward:   out = block(x) + shortcut(x)
 * Backward:  grad splits additively to both paths
 *
 * NOTE: This implementation only handles the 2D [batch, features] case
 * through Linear layers.  For Conv2D residual blocks the user can
 * compose layers manually with the same add-input pattern.
 */

#include "CppNet/layers/residual.hpp"
#include <stdexcept>

namespace CppNet
{
    namespace Layers
    {
        Residual::Residual(std::vector<std::shared_ptr<Layer>> block,
                           int in_features, int out_features,
                           const std::string& device)
            : block_(std::move(block)),
              in_features_(in_features), out_features_(out_features),
              device_(device)
        {
            if (block_.empty())
                throw std::invalid_argument("Residual: block must contain at least one layer");

            // Create a projection shortcut if dimensions differ
            if (in_features_ != out_features_)
            {
                projection_ = std::make_shared<Linear>(
                    in_features_, out_features_, "ResProj",
                    /*trainable=*/true, /*bias=*/false, device);
            }
        }

        Residual::~Residual() = default;

        const Eigen::Tensor<float, 2> Residual::forward(
            const Eigen::Tensor<float, 2>& input)
        {
            input_cache_ = input;

            // --- Main path: run through the block ---
            // We dynamic_cast to Linear to call the typed forward().
            // A more generic approach would use a variant / type-erased forward,
            // but for the current CppNet architecture this is the practical path.
            Eigen::Tensor<float, 2> out = input;
            for (auto& layer : block_)
            {
                auto* lin = dynamic_cast<Linear*>(layer.get());
                if (lin)
                {
                    out = lin->forward(out);
                    continue;
                }
                // If it's not a Linear, we skip (activations should be
                // applied inline by the user or via a thin wrapper).
            }

            // --- Shortcut path ---
            Eigen::Tensor<float, 2> shortcut;
            if (projection_)
                shortcut = projection_->forward(input);
            else
                shortcut = input;

            // --- Element-wise addition ---
            int rows = out.dimension(0);
            int cols = out.dimension(1);
            Eigen::Tensor<float, 2> result(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result(i, j) = out(i, j) + shortcut(i, j);

            return result;
        }

        const Eigen::Tensor<float, 2> Residual::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            // Gradient flows identically to both the block and the shortcut.

            // --- Block backward (reverse order) ---
            Eigen::Tensor<float, 2> grad_block = grad_output;
            for (int i = static_cast<int>(block_.size()) - 1; i >= 0; --i)
            {
                auto* lin = dynamic_cast<Linear*>(block_[i].get());
                if (lin)
                    grad_block = lin->backward(grad_block);
            }

            // --- Shortcut backward ---
            Eigen::Tensor<float, 2> grad_shortcut;
            if (projection_)
                grad_shortcut = projection_->backward(grad_output);
            else
                grad_shortcut = grad_output;

            // Sum gradients from both paths
            int rows = grad_block.dimension(0);
            int cols = grad_block.dimension(1);
            Eigen::Tensor<float, 2> grad_input(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    grad_input(i, j) = grad_block(i, j) + grad_shortcut(i, j);

            return grad_input;
        }

        void Residual::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            for (auto& layer : block_)
                if (layer->is_trainable())
                    layer->step(optimizer, learning_rate);

            if (projection_)
                projection_->step(optimizer, learning_rate);
        }
    }
}
