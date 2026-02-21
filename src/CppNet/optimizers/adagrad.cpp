/**
 * @file adagrad.cpp
 * @brief Adagrad optimizer implementation
 *
 * G += dW^2
 * W -= lr * dW / (sqrt(G) + eps)
 */

#include "CppNet/optimizers/adagrad.hpp"
#include <cmath>

namespace CppNet
{
    namespace Optimizers
    {
        Adagrad::Adagrad(float epsilon)
            : epsilon_(epsilon)
        {
        }

        void Adagrad::step(CppNet::Layers::Linear& layer, float learning_rate)
        {
            if (!layer.is_trainable()) return;

            void* key = static_cast<void*>(&layer);

            auto& weights = layer.get_weights();
            auto& grad_w = layer.get_grad_weights();
            int w_rows = weights.dimension(0);
            int w_cols = weights.dimension(1);

            if (accum_weights_.find(key) == accum_weights_.end())
            {
                accum_weights_[key] = Eigen::Tensor<float, 2>(w_rows, w_cols);
                accum_weights_[key].setZero();
            }

            auto& g_w = accum_weights_[key];

            for (int i = 0; i < w_rows; ++i)
                for (int j = 0; j < w_cols; ++j)
                {
                    g_w(i, j) += grad_w(i, j) * grad_w(i, j);
                    weights(i, j) -= learning_rate * grad_w(i, j) / (std::sqrt(g_w(i, j)) + epsilon_);
                }

            if (layer.has_bias())
            {
                auto& biases = layer.get_biases();
                auto& grad_b = layer.get_grad_biases();
                int b_size = biases.dimension(0);

                if (accum_biases_.find(key) == accum_biases_.end())
                {
                    accum_biases_[key] = Eigen::Tensor<float, 1>(b_size);
                    accum_biases_[key].setZero();
                }

                auto& g_b = accum_biases_[key];

                for (int i = 0; i < b_size; ++i)
                {
                    g_b(i) += grad_b(i) * grad_b(i);
                    biases(i) -= learning_rate * grad_b(i) / (std::sqrt(g_b(i)) + epsilon_);
                }
            }
        }
    }
}
