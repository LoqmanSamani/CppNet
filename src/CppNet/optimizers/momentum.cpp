/**
 * @file momentum.cpp
 * @brief SGD with Momentum optimizer implementation
 *
 * v = mu * v - lr * dW
 * W += v
 */

#include "CppNet/optimizers/momentum.hpp"

namespace CppNet
{
    namespace Optimizers
    {
        Momentum::Momentum(float mu)
            : mu_(mu)
        {
        }

        void Momentum::step(CppNet::Layers::Linear& layer, float learning_rate)
        {
            if (!layer.is_trainable()) return;

            void* key = static_cast<void*>(&layer);

            auto& weights = layer.get_weights();
            auto& grad_w = layer.get_grad_weights();
            int w_rows = weights.dimension(0);
            int w_cols = weights.dimension(1);

            if (velocity_weights_.find(key) == velocity_weights_.end())
            {
                velocity_weights_[key] = Eigen::Tensor<float, 2>(w_rows, w_cols);
                velocity_weights_[key].setZero();
            }

            auto& v_w = velocity_weights_[key];

            for (int i = 0; i < w_rows; ++i)
                for (int j = 0; j < w_cols; ++j)
                {
                    v_w(i, j) = mu_ * v_w(i, j) - learning_rate * grad_w(i, j);
                    weights(i, j) += v_w(i, j);
                }

            if (layer.has_bias())
            {
                auto& biases = layer.get_biases();
                auto& grad_b = layer.get_grad_biases();
                int b_size = biases.dimension(0);

                if (velocity_biases_.find(key) == velocity_biases_.end())
                {
                    velocity_biases_[key] = Eigen::Tensor<float, 1>(b_size);
                    velocity_biases_[key].setZero();
                }

                auto& v_b = velocity_biases_[key];

                for (int i = 0; i < b_size; ++i)
                {
                    v_b(i) = mu_ * v_b(i) - learning_rate * grad_b(i);
                    biases(i) += v_b(i);
                }
            }
        }
    }
}
