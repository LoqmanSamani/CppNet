/**
 * @file adam.hpp
 * @brief Adam optimizer
 */

#ifndef ADAM_HPP
#define ADAM_HPP

#include "CppNet/optimizers/optimizer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unordered_map>
#include <vector>
#include <string>

namespace CppNet
{
    namespace Optimizers
    {
        /**
         * @class Adam
         * @brief Adam optimizer: adaptive learning rates with momentum.
         *
         * Maintains per-parameter first (m) and second (v) moment estimates.
         * Update rule: W -= lr * m_hat / (sqrt(v_hat) + eps)
         */
        class Adam : public Optimizer
        {
        public:
            /**
             * @param beta1 Exponential decay rate for first moment (default 0.9)
             * @param beta2 Exponential decay rate for second moment (default 0.999)
             * @param epsilon Small constant for numerical stability (default 1e-8)
             */
            Adam(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

            void update(float* weights, const float* gradients,
                        int size, float learning_rate) override;

        private:
            float beta1_;
            float beta2_;
            float epsilon_;
            int t_ = 0;  // timestep counter
            std::unordered_map<void*, std::vector<float>> m_params_;
            std::unordered_map<void*, std::vector<float>> v_params_;
            std::unordered_map<void*, int> t_params_;
        };
    }
}

#endif // ADAM_HPP
