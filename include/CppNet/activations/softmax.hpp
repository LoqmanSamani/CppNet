/**
 * @file softmax.hpp
 * @brief Softmax activation function
 */

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "CppNet/activations/activation.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Activations
    {
        /**
         * @class Softmax
         * @brief Softmax activation: softmax(x_i) = exp(x_i) / sum(exp(x_j))
         *
         * Applies softmax along the last dimension. Forward normalizes
         * inputs to a probability distribution; backward computes the
         * Jacobian-vector product.
         */
        class Softmax final : public Activation
        {
        public:
            /**
             * @param device Compute backend: "cpu-eigen", "cpu", or "gpu"
             */
            explicit Softmax(const std::string& device = "cpu-eigen");

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

#endif // SOFTMAX_HPP

