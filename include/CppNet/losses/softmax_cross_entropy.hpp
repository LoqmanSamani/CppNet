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

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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
            explicit SoftmaxCrossEntropy(const std::string& reduction = "mean",
                                         const std::string& device = "cpu");
            ~SoftmaxCrossEntropy() override;

            float forward(const Eigen::Tensor<float, 2>& logits,
                          const Eigen::Tensor<float, 2>& targets) override;

            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& logits,
                                              const Eigen::Tensor<float, 2>& targets) override;

        private:
            std::string reduction_;
            std::string device_;
            Eigen::Tensor<float, 2> softmax_cache_;

            #ifdef USE_CUDA
                float* d_logits_ = nullptr;
                float* d_targets_ = nullptr;
                float* d_softmax_ = nullptr;
                float* d_loss_ = nullptr;
                float* d_grad_ = nullptr;
                std::size_t gpu_buf_ = 0;
                bool gpu_init_ = false;
                void ensure_gpu(std::size_t n);
                void release_gpu();
                float forward_gpu(const Eigen::Tensor<float, 2>& logits,
                                  const Eigen::Tensor<float, 2>& targets);
                void backward_gpu(const Eigen::Tensor<float, 2>& targets,
                                  Eigen::Tensor<float, 2>& grad);
            #endif
        };
    }
}

#endif // SOFTMAX_CROSS_ENTROPY_HPP
