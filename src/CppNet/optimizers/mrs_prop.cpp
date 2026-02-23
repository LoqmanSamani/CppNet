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
    }
}
