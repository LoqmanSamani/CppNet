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

        void Momentum::update(float* weights, const float* gradients,
                              int size, float learning_rate)
        {
            void* key = static_cast<void*>(weights);

            if (vel_params_.find(key) == vel_params_.end())
                vel_params_[key].assign(size, 0.0f);

            auto& vel = vel_params_[key];

            for (int i = 0; i < size; ++i)
            {
                vel[i] = mu_ * vel[i] - learning_rate * gradients[i];
                weights[i] += vel[i];
            }
        }
    }
}
