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

        void Adam::update(float* weights, const float* gradients,
                          int size, float learning_rate)
        {
            void* key = static_cast<void*>(weights);

            if (m_params_.find(key) == m_params_.end())
            {
                m_params_[key].assign(size, 0.0f);
                v_params_[key].assign(size, 0.0f);
                t_params_[key] = 0;
            }

            auto& m = m_params_[key];
            auto& v = v_params_[key];
            int& t = t_params_[key];
            ++t;

            float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t));
            float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t));

            for (int i = 0; i < size; ++i)
            {
                m[i] = beta1_ * m[i] + (1.0f - beta1_) * gradients[i];
                v[i] = beta2_ * v[i] + (1.0f - beta2_) * gradients[i] * gradients[i];
                float mh = m[i] / bc1;
                float vh = v[i] / bc2;
                weights[i] -= learning_rate * mh / (std::sqrt(vh) + epsilon_);
            }
        }
    }
}
