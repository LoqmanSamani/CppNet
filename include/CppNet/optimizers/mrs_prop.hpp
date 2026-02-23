/**
 * @file mrs_prop.hpp
 * @brief RMSProp optimizer
 */

#ifndef RMS_PROP_HPP
#define RMS_PROP_HPP

#include "CppNet/optimizers/optimizer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unordered_map>
#include <vector>

namespace CppNet
{
    namespace Optimizers
    {
        /**
         * @class RMSProp
         * @brief RMSProp: divides learning rate by a running avg of gradient magnitudes.
         *
         * Update: s = rho * s + (1-rho) * dW^2; W -= lr * dW / (sqrt(s) + eps)
         */
        class RMSProp : public Optimizer
        {
        public:
            RMSProp(float rho = 0.9f, float epsilon = 1e-8f);

            void update(float* weights, const float* gradients,
                        int size, float learning_rate) override;

        private:
            float rho_;
            float epsilon_;
            std::unordered_map<void*, std::vector<float>> cache_params_;
        };
    }
}

#endif // RMS_PROP_HPP
