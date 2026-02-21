/**
 * @file regularizations.cpp
 * @brief Implementation of L1, L2, and Elastic Net regularization
 */

#include "CppNet/regularizations/regularizations.hpp"
#include <cmath>

namespace CppNet
{
    namespace Regularizations
    {
        float l1_penalty(const Eigen::Tensor<float, 2>& weights, float lambda)
        {
            // L1 penalty = lambda * sum(|W|)
            Eigen::Tensor<float, 0> sum = weights.abs().sum();
            return lambda * sum(0);
        }

        Eigen::Tensor<float, 2> l1_gradient(const Eigen::Tensor<float, 2>& weights, float lambda)
        {
            // L1 gradient = lambda * sign(W)
            return weights.sign() * weights.constant(lambda);
        }

        float l2_penalty(const Eigen::Tensor<float, 2>& weights, float lambda)
        {
            // L2 penalty = 0.5 * lambda * sum(W^2)
            Eigen::Tensor<float, 0> sum = weights.square().sum();
            return 0.5f * lambda * sum(0);
        }

        Eigen::Tensor<float, 2> l2_gradient(const Eigen::Tensor<float, 2>& weights, float lambda)
        {
            // L2 gradient = lambda * W
            return weights * weights.constant(lambda);
        }

        float elastic_net_penalty(const Eigen::Tensor<float, 2>& weights, float lambda, float l1_ratio)
        {
            // ElasticNet = l1_ratio * L1 + (1 - l1_ratio) * L2
            return l1_ratio * l1_penalty(weights, lambda) +
                   (1.0f - l1_ratio) * l2_penalty(weights, lambda);
        }

        Eigen::Tensor<float, 2> elastic_net_gradient(const Eigen::Tensor<float, 2>& weights, float lambda, float l1_ratio)
        {
            return l1_gradient(weights, lambda * l1_ratio) +
                   l2_gradient(weights, lambda * (1.0f - l1_ratio));
        }
    }
}
