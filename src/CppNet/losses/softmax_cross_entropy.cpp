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
#include "CppNet/kernels/gpu/gpu.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace CppNet
{
    namespace Losses
    {
        SoftmaxCrossEntropy::SoftmaxCrossEntropy(const std::string& reduction,
                                                 const std::string& device)
            : reduction_(reduction), device_(device)
        {
            if (reduction_ != "mean" && reduction_ != "sum")
                throw std::invalid_argument(
                    "SoftmaxCrossEntropy: reduction must be \"mean\" or \"sum\"");
        }

        SoftmaxCrossEntropy::~SoftmaxCrossEntropy()
        {
            #ifdef USE_CUDA
                release_gpu();
            #endif
        }

        #ifdef USE_CUDA

            void SoftmaxCrossEntropy::ensure_gpu(std::size_t n)
            {
                if (gpu_init_ && gpu_buf_ >= n) return;
                release_gpu();
                cudaMalloc(&d_logits_, n * sizeof(float));
                cudaMalloc(&d_targets_, n * sizeof(float));
                cudaMalloc(&d_softmax_, n * sizeof(float));
                cudaMalloc(&d_loss_, sizeof(float));
                cudaMalloc(&d_grad_, n * sizeof(float));
                gpu_buf_ = n;
                gpu_init_ = true;
            }

            void SoftmaxCrossEntropy::release_gpu()
            {
                if (d_logits_)  cudaFree(d_logits_);
                if (d_targets_) cudaFree(d_targets_);
                if (d_softmax_) cudaFree(d_softmax_);
                if (d_loss_)    cudaFree(d_loss_);
                if (d_grad_)    cudaFree(d_grad_);
                d_logits_ = nullptr; d_targets_ = nullptr;
                d_softmax_ = nullptr; d_loss_ = nullptr; d_grad_ = nullptr;
                gpu_buf_ = 0; gpu_init_ = false;
            }

            float SoftmaxCrossEntropy::forward_gpu(const Eigen::Tensor<float, 2>& logits,
                                                   const Eigen::Tensor<float, 2>& targets)
            {
                int batch = logits.dimension(0);
                int classes = logits.dimension(1);
                std::size_t n = logits.size();
                ensure_gpu(n);

                cudaMemcpy(d_logits_, logits.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_targets_, targets.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                float zero = 0.0f;
                cudaMemcpy(d_loss_, &zero, sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int smem = block * sizeof(float);
                Kernels::GPU::softmax_ce_forward_kernel<<<batch, block, smem>>>(
                    d_logits_, d_targets_, d_softmax_, d_loss_, batch, classes);

                float loss;
                cudaMemcpy(&loss, d_loss_, sizeof(float), cudaMemcpyDeviceToHost);

                // download softmax cache for backward
                softmax_cache_.resize(batch, classes);
                cudaMemcpy(softmax_cache_.data(), d_softmax_, n * sizeof(float), cudaMemcpyDeviceToHost);

                if (reduction_ == "mean")
                    return loss / static_cast<float>(batch);
                return loss;
            }

            void SoftmaxCrossEntropy::backward_gpu(const Eigen::Tensor<float, 2>& targets,
                                                   Eigen::Tensor<float, 2>& grad)
            {
                int batch = targets.dimension(0);
                int classes = targets.dimension(1);
                std::size_t n = targets.size();

                // softmax is already on device from forward, but re-upload to be safe
                cudaMemcpy(d_softmax_, softmax_cache_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_targets_, targets.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                float scale = (reduction_ == "mean") ? 1.0f / static_cast<float>(batch) : 1.0f;

                int block = 256;
                int grid = (static_cast<int>(n) + block - 1) / block;
                Kernels::GPU::softmax_ce_backward_kernel<<<grid, block>>>(
                    d_softmax_, d_targets_, d_grad_, static_cast<int>(n), scale);

                cudaMemcpy(grad.data(), d_grad_, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

        #endif

        float SoftmaxCrossEntropy::forward(
            const Eigen::Tensor<float, 2>& logits,
            const Eigen::Tensor<float, 2>& targets)
        {
            int batch = logits.dimension(0);
            int classes = logits.dimension(1);

            #ifdef USE_CUDA
                if (device_ == "gpu") return forward_gpu(logits, targets);
            #endif

            softmax_cache_.resize(batch, classes);
            float total_loss = 0.0f;

            for (int n = 0; n < batch; ++n)
            {
                float max_val = logits(n, 0);
                for (int c = 1; c < classes; ++c)
                    max_val = std::max(max_val, logits(n, c));

                float sum_exp = 0.0f;
                for (int c = 0; c < classes; ++c)
                {
                    softmax_cache_(n, c) = std::exp(logits(n, c) - max_val);
                    sum_exp += softmax_cache_(n, c);
                }

                float log_sum = std::log(sum_exp + 1e-12f);
                for (int c = 0; c < classes; ++c)
                {
                    softmax_cache_(n, c) /= sum_exp;
                    float log_softmax = (logits(n, c) - max_val) - log_sum;
                    total_loss -= targets(n, c) * log_softmax;
                }
            }

            if (reduction_ == "mean")
                return total_loss / static_cast<float>(batch);
            return total_loss;
        }

        Eigen::Tensor<float, 2> SoftmaxCrossEntropy::backward(
            const Eigen::Tensor<float, 2>& logits,
            const Eigen::Tensor<float, 2>& targets)
        {
            int batch = logits.dimension(0);
            int classes = logits.dimension(1);

            if (softmax_cache_.dimension(0) != batch ||
                softmax_cache_.dimension(1) != classes)
            {
                forward(logits, targets);
            }

            Eigen::Tensor<float, 2> grad(batch, classes);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    backward_gpu(targets, grad);
                    return grad;
                }
            #endif

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
