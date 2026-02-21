/**
 * @file momentum.hpp
 * @brief SGD with Momentum optimizer
 */

#ifndef MOMENTUM_HPP
#define MOMENTUM_HPP

#include "CppNet/optimizers/optimizer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unordered_map>

namespace CppNet
{
    namespace Optimizers
    {
        /**
         * @class Momentum
         * @brief SGD with Momentum: v = mu * v - lr * dW; W += v
         */
        class Momentum : public Optimizer
        {
        public:
            explicit Momentum(float mu = 0.9f);

            void step(CppNet::Layers::Linear& layer, float learning_rate) override;

        private:
            float mu_;
            std::unordered_map<void*, Eigen::Tensor<float, 2>> velocity_weights_;
            std::unordered_map<void*, Eigen::Tensor<float, 1>> velocity_biases_;
        };
    }
}

#endif // MOMENTUM_HPP
