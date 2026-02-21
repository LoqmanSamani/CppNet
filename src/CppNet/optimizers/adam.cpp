/**
 * @file adam.cpp
 * @brief Adam optimizer implementation
 *
 * m = beta1 * m + (1 - beta1) * dW
 * v = beta2 * v + (1 - beta2) * dW^2
 * m_hat = m / (1 - beta1^t)
 * v_hat = v / (1 - beta2^t)
 * W -= lr * m_hat / (sqrt(v_hat) + eps)
 */

#include "CppNet/optimizers/adam.hpp"
#include <cmath>

namespace CppNet
{
    namespace Optimizers
    {
        Adam::Adam(float beta1, float beta2, float epsilon)
            : beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0)
        {
        }

        void Adam::step(CppNet::Layers::Linear& layer, float learning_rate)
        {
            if (!layer.is_trainable()) return;

            ++t_;

            void* key = static_cast<void*>(&layer);

            auto& weights = layer.get_weights();
            auto& grad_w = layer.get_grad_weights();
            int w_rows = weights.dimension(0);
            int w_cols = weights.dimension(1);

            // Initialize moment caches if first time
            if (m_weights_.find(key) == m_weights_.end())
            {
                m_weights_[key] = Eigen::Tensor<float, 2>(w_rows, w_cols);
                m_weights_[key].setZero();
                v_weights_[key] = Eigen::Tensor<float, 2>(w_rows, w_cols);
                v_weights_[key].setZero();
            }

            auto& m_w = m_weights_[key];
            auto& v_w = v_weights_[key];

            float beta1_t = std::pow(beta1_, static_cast<float>(t_));
            float beta2_t = std::pow(beta2_, static_cast<float>(t_));

            // Update weight moments and apply
            for (int i = 0; i < w_rows; ++i)
            {
                for (int j = 0; j < w_cols; ++j)
                {
                    m_w(i, j) = beta1_ * m_w(i, j) + (1.0f - beta1_) * grad_w(i, j);
                    v_w(i, j) = beta2_ * v_w(i, j) + (1.0f - beta2_) * grad_w(i, j) * grad_w(i, j);

                    float m_hat = m_w(i, j) / (1.0f - beta1_t);
                    float v_hat = v_w(i, j) / (1.0f - beta2_t);

                    weights(i, j) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon_);
                }
            }

            // Update bias if present
            if (layer.has_bias())
            {
                auto& biases = layer.get_biases();
                auto& grad_b = layer.get_grad_biases();
                int b_size = biases.dimension(0);

                if (m_biases_.find(key) == m_biases_.end())
                {
                    m_biases_[key] = Eigen::Tensor<float, 1>(b_size);
                    m_biases_[key].setZero();
                    v_biases_[key] = Eigen::Tensor<float, 1>(b_size);
                    v_biases_[key].setZero();
                }

                auto& m_b = m_biases_[key];
                auto& v_b = v_biases_[key];

                for (int i = 0; i < b_size; ++i)
                {
                    m_b(i) = beta1_ * m_b(i) + (1.0f - beta1_) * grad_b(i);
                    v_b(i) = beta2_ * v_b(i) + (1.0f - beta2_) * grad_b(i) * grad_b(i);

                    float m_hat = m_b(i) / (1.0f - beta1_t);
                    float v_hat = v_b(i) / (1.0f - beta2_t);

                    biases(i) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon_);
                }
            }
        }
    }
}
