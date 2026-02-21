/**
 * @file adagrad.hpp
 * @brief Adagrad optimizer
 */

#ifndef ADAGRAD_HPP
#define ADAGRAD_HPP

#include "CppNet/optimizers/optimizer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unordered_map>

namespace CppNet
{
    namespace Optimizers
    {
        /**
         * @class Adagrad
         * @brief Adagrad optimizer: scales learning rate by accumulated gradient squares.
         *
         * Update rule: W -= lr * dW / (sqrt(G) + eps), where G accumulates dW^2.
         */
        class Adagrad : public Optimizer
        {
        public:
            explicit Adagrad(float epsilon = 1e-8f);

            void step(CppNet::Layers::Linear& layer, float learning_rate) override;

        private:
            float epsilon_;
            std::unordered_map<void*, Eigen::Tensor<float, 2>> accum_weights_;
            std::unordered_map<void*, Eigen::Tensor<float, 1>> accum_biases_;
        };
    }
}

#endif // ADAGRAD_HPP
