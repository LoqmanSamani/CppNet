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

        void Adagrad::update(float* weights, const float* gradients,
                             int size, float learning_rate)
        {
            void* key = static_cast<void*>(weights);

            if (accum_params_.find(key) == accum_params_.end())
                accum_params_[key].assign(size, 0.0f);

            auto& g = accum_params_[key];

            for (int i = 0; i < size; ++i)
            {
                g[i] += gradients[i] * gradients[i];
                weights[i] -= learning_rate * gradients[i] / (std::sqrt(g[i]) + epsilon_);
            }
        }
    }
}
