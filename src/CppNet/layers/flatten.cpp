/**
 * @file flatten.cpp
 * @brief Flatten layer implementation
 *
 * Reshapes [batch, C, H, W] -> [batch, C*H*W] on forward,
 * and reverses the reshape on backward.
 */

#include "CppNet/layers/flatten.hpp"

namespace CppNet
{
    namespace Layers
    {
        const Eigen::Tensor<float, 2> Flatten::forward(const Eigen::Tensor<float, 4>& input)
        {
            input_shape_ = {
                static_cast<int>(input.dimension(0)),
                static_cast<int>(input.dimension(1)),
                static_cast<int>(input.dimension(2)),
                static_cast<int>(input.dimension(3))
            };

            int batch = input_shape_[0];
            int flat_size = input_shape_[1] * input_shape_[2] * input_shape_[3];

            // reshape 4D -> 2D
            Eigen::array<Eigen::Index, 2> dims = {batch, flat_size};
            return input.reshape(dims);
        }

        const Eigen::Tensor<float, 4> Flatten::backward(const Eigen::Tensor<float, 2>& grad_output)
        {
            // reshape 2D -> 4D back to original shape
            Eigen::array<Eigen::Index, 4> dims = {
                input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]
            };
            return grad_output.reshape(dims);
        }
    }
}
