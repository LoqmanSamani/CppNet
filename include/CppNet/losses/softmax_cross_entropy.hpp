/**
 * @file softmax_cross_entropy.hpp
 * @brief Fused Softmax + Cross-Entropy loss
 *
 * Combines log-softmax and negative log-likelihood into a single,
 * numerically stable operation.
 *
 * Forward:  L = -1/N * sum_i [ target_i * log_softmax(logit_i) ]
 *
 * Backward: dL/d(logit) = softmax(logit) - target  (one-hot)
 *
 * This avoids the numerical instability of computing softmax first
 * and then taking log(), and produces a simpler gradient.
 */

#ifndef SOFTMAX_CROSS_ENTROPY_HPP
#define SOFTMAX_CROSS_ENTROPY_HPP

#include "CppNet/losses/loss.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Losses
    {
        /**
         * @class SoftmaxCrossEntropy
         * @brief Fused log-softmax + NLL loss
         *
         * Input logits:  [batch, num_classes]  (raw, unnormalized)
         * Targets:        [batch, num_classes]  (one-hot encoded)
         */
        class SoftmaxCrossEntropy : public Loss
        {
        public:
            /**
             * @param reduction "mean" (default) or "sum"
             */
            explicit SoftmaxCrossEntropy(const std::string& reduction = "mean");
            ~SoftmaxCrossEntropy() override = default;

            float forward(const Eigen::Tensor<float, 2>& logits,
                          const Eigen::Tensor<float, 2>& targets) override;

            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& logits,
                                              const Eigen::Tensor<float, 2>& targets) override;

        private:
            std::string reduction_;

            /// Cached softmax output from forward (reused in backward)
            Eigen::Tensor<float, 2> softmax_cache_;
        };
    }
}

#endif // SOFTMAX_CROSS_ENTROPY_HPP
