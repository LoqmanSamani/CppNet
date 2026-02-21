/**
 * @file softmax_cross_entropy.cpp
 * @brief Fused Softmax + Cross-Entropy implementation
 *
 * Numerically stable log-softmax trick:
 *   log_softmax_j = logit_j - max_i(logit_i) - log(sum_i exp(logit_i - max))
 *
 * Loss = -1/N * sum_{n,c} target[n,c] * log_softmax[n,c]
 * Grad = (softmax - target) / N        (when reduction == "mean")
 */

#include "CppNet/losses/softmax_cross_entropy.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace CppNet
{
    namespace Losses
    {
        SoftmaxCrossEntropy::SoftmaxCrossEntropy(const std::string& reduction)
            : reduction_(reduction)
        {
            if (reduction_ != "mean" && reduction_ != "sum")
                throw std::invalid_argument(
                    "SoftmaxCrossEntropy: reduction must be \"mean\" or \"sum\"");
        }

        float SoftmaxCrossEntropy::forward(
            const Eigen::Tensor<float, 2>& logits,
            const Eigen::Tensor<float, 2>& targets)
        {
            int batch = logits.dimension(0);
            int classes = logits.dimension(1);

            softmax_cache_.resize(batch, classes);
            float total_loss = 0.0f;

            for (int n = 0; n < batch; ++n)
            {
                // 1. Find max for numerical stability
                float max_val = logits(n, 0);
                for (int c = 1; c < classes; ++c)
                    max_val = std::max(max_val, logits(n, c));

                // 2. Compute exp(logit - max) and their sum
                float sum_exp = 0.0f;
                for (int c = 0; c < classes; ++c)
                {
                    softmax_cache_(n, c) = std::exp(logits(n, c) - max_val);
                    sum_exp += softmax_cache_(n, c);
                }

                // 3. Normalize to get softmax, accumulate loss
                float log_sum = std::log(sum_exp + 1e-12f);
                for (int c = 0; c < classes; ++c)
                {
                    softmax_cache_(n, c) /= sum_exp;
                    // log_softmax = (logit - max) - log(sum_exp)
                    float log_softmax = (logits(n, c) - max_val) - log_sum;
                    total_loss -= targets(n, c) * log_softmax;
                }
            }

            if (reduction_ == "mean")
                return total_loss / static_cast<float>(batch);
            return total_loss;  // "sum"
        }

        Eigen::Tensor<float, 2> SoftmaxCrossEntropy::backward(
            const Eigen::Tensor<float, 2>& logits,
            const Eigen::Tensor<float, 2>& targets)
        {
            // If forward() hasn't been called yet, compute softmax now
            int batch = logits.dimension(0);
            int classes = logits.dimension(1);

            if (softmax_cache_.dimension(0) != batch ||
                softmax_cache_.dimension(1) != classes)
            {
                forward(logits, targets);  // populate cache
            }

            Eigen::Tensor<float, 2> grad(batch, classes);

            for (int n = 0; n < batch; ++n)
                for (int c = 0; c < classes; ++c)
                    grad(n, c) = softmax_cache_(n, c) - targets(n, c);

            if (reduction_ == "mean")
            {
                float inv_batch = 1.0f / static_cast<float>(batch);
                for (int n = 0; n < batch; ++n)
                    for (int c = 0; c < classes; ++c)
                        grad(n, c) *= inv_batch;
            }

            return grad;
        }
    }
}
