/**
 * @file regularizations.hpp
 * @brief Regularization techniques for CppNet
 *
 * Provides L1, L2, and Elastic Net regularization penalties
 * that can be applied to layer weights during training.
 */

#ifndef REGULARIZATIONS_HPP
#define REGULARIZATIONS_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace CppNet
{
    namespace Regularizations
    {
        /**
         * @brief Compute L1 regularization penalty
         * @param weights Weight tensor [in_size, out_size]
         * @param lambda Regularization strength
         * @return Scalar penalty value (lambda * sum(|W|))
         */
        float l1_penalty(const Eigen::Tensor<float, 2>& weights, float lambda);

        /**
         * @brief Compute L1 regularization gradient
         * @param weights Weight tensor [in_size, out_size]
         * @param lambda Regularization strength
         * @return Gradient tensor (lambda * sign(W))
         */
        Eigen::Tensor<float, 2> l1_gradient(const Eigen::Tensor<float, 2>& weights, float lambda);

        /**
         * @brief Compute L2 regularization penalty
         * @param weights Weight tensor [in_size, out_size]
         * @param lambda Regularization strength
         * @return Scalar penalty value (0.5 * lambda * sum(W^2))
         */
        float l2_penalty(const Eigen::Tensor<float, 2>& weights, float lambda);

        /**
         * @brief Compute L2 regularization gradient
         * @param weights Weight tensor [in_size, out_size]
         * @param lambda Regularization strength
         * @return Gradient tensor (lambda * W)
         */
        Eigen::Tensor<float, 2> l2_gradient(const Eigen::Tensor<float, 2>& weights, float lambda);

        /**
         * @brief Compute Elastic Net penalty (L1 + L2 combined)
         * @param weights Weight tensor
         * @param lambda Total regularization strength
         * @param l1_ratio Fraction allocated to L1 (0 = all L2, 1 = all L1)
         * @return Scalar penalty value
         */
        float elastic_net_penalty(const Eigen::Tensor<float, 2>& weights, float lambda, float l1_ratio = 0.5f);

        /**
         * @brief Compute Elastic Net gradient
         * @param weights Weight tensor
         * @param lambda Total regularization strength
         * @param l1_ratio Fraction allocated to L1
         * @return Gradient tensor
         */
        Eigen::Tensor<float, 2> elastic_net_gradient(const Eigen::Tensor<float, 2>& weights, float lambda, float l1_ratio = 0.5f);
    }
}

#endif // REGULARIZATIONS_HPP
