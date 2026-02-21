/**
 * @file tanh.hpp
 * @brief Tanh activation function
 */

#ifndef TANH_ACTIVATION_HPP
#define TANH_ACTIVATION_HPP

#include "CppNet/activations/activation.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Activations
    {
        /**
         * @class Tanh
         * @brief Tanh activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
         *
         * Output range is (-1, 1). Backward: grad * (1 - tanh(x)^2).
         */
        class Tanh final : public Activation
        {
        public:
            explicit Tanh(const std::string& device = "cpu-eigen");

            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_activation) override;
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) override;
            Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_activation) override;
            Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output) override;

            static void set_num_threads(int num_threads);

        private:
            std::string device_;
            Eigen::Tensor<float, 2> output_cache_2d_;
            Eigen::Tensor<float, 4> output_cache_4d_;
        };
    }
}

#endif // TANH_ACTIVATION_HPP
