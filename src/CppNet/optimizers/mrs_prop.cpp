/**
 * @file mrs_prop.cpp
 * @brief RMSProp optimizer implementation
 *
 * s = rho * s + (1 - rho) * dW^2
 * W -= lr * dW / (sqrt(s) + eps)
 */

#include "CppNet/optimizers/mrs_prop.hpp"
#include <cmath>

namespace CppNet
{
    namespace Optimizers
    {
        RMSProp::RMSProp(float rho, float epsilon)
            : rho_(rho), epsilon_(epsilon)
        {
        }

        void RMSProp::update(float* weights, const float* gradients,
                             int size, float learning_rate)
        {
            void* key = static_cast<void*>(weights);

            if (cache_params_.find(key) == cache_params_.end())
                cache_params_[key].assign(size, 0.0f);

            auto& s = cache_params_[key];

            for (int i = 0; i < size; ++i)
            {
                s[i] = rho_ * s[i] + (1.0f - rho_) * gradients[i] * gradients[i];
                weights[i] -= learning_rate * gradients[i] / (std::sqrt(s[i]) + epsilon_);
            }
        }

        void RMSProp::step(CppNet::Layers::Linear& layer, float learning_rate)
        {
            if (!layer.is_trainable()) return;

            void* key = static_cast<void*>(&layer);

            auto& weights = layer.get_weights();
            auto& grad_w = layer.get_grad_weights();
            int w_rows = weights.dimension(0);
            int w_cols = weights.dimension(1);

            if (cache_weights_.find(key) == cache_weights_.end())
            {
                cache_weights_[key] = Eigen::Tensor<float, 2>(w_rows, w_cols);
                cache_weights_[key].setZero();
            }

            auto& s_w = cache_weights_[key];

            for (int i = 0; i < w_rows; ++i)
                for (int j = 0; j < w_cols; ++j)
                {
                    s_w(i, j) = rho_ * s_w(i, j) + (1.0f - rho_) * grad_w(i, j) * grad_w(i, j);
                    weights(i, j) -= learning_rate * grad_w(i, j) / (std::sqrt(s_w(i, j)) + epsilon_);
                }

            if (layer.has_bias())
            {
                auto& biases = layer.get_biases();
                auto& grad_b = layer.get_grad_biases();
                int b_size = biases.dimension(0);

                if (cache_biases_.find(key) == cache_biases_.end())
                {
                    cache_biases_[key] = Eigen::Tensor<float, 1>(b_size);
                    cache_biases_[key].setZero();
                }

                auto& s_b = cache_biases_[key];

                for (int i = 0; i < b_size; ++i)
                {
                    s_b(i) = rho_ * s_b(i) + (1.0f - rho_) * grad_b(i) * grad_b(i);
                    biases(i) -= learning_rate * grad_b(i) / (std::sqrt(s_b(i)) + epsilon_);
                }
            }
        }
    }
}
