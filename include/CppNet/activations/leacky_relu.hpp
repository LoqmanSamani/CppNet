/**
 * @file leacky_relu.hpp
 * @brief Leaky ReLU activation function
 */

#ifndef LEAKY_RELU_HPP
#define LEAKY_RELU_HPP

#include "CppNet/activations/activation.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Activations
    {
        /**
         * @class LeakyReLU
         * @brief Leaky ReLU: f(x) = x if x > 0, else alpha * x
         *
         * Prevents "dying ReLU" problem by allowing small negative gradients.
         */
        class LeakyReLU final : public Activation
        {
        public:
            /**
             * @param alpha Slope for negative inputs (default 0.01)
             * @param device Compute backend: "cpu-eigen", "cpu", or "gpu"
             */
            explicit LeakyReLU(float alpha = 0.01f, const std::string& device = "cpu-eigen");

            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_activation) override;
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) override;
            Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_activation) override;
            Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output) override;

            static void set_num_threads(int num_threads);
            float get_alpha() const { return alpha_; }

        private:
            float alpha_;
            std::string device_;
            Eigen::Tensor<float, 2> input_cache_2d_;
            Eigen::Tensor<float, 4> input_cache_4d_;
        };
    }
}

#endif // LEAKY_RELU_HPP

