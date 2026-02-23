/**
 * @file batch_norm.cpp
 * @brief Batch Normalization layer implementation
 *
 * Training:   x_hat = (x - mean) / sqrt(var + eps)
 *             y = gamma * x_hat + beta
 *             running_mean = (1-m)*running_mean + m*batch_mean
 *             running_var  = (1-m)*running_var  + m*batch_var
 *
 * Inference:  x_hat = (x - running_mean) / sqrt(running_var + eps)
 *             y = gamma * x_hat + beta
 */

#include "CppNet/layers/batch_norm.hpp"
#include <cmath>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Layers
    {
        BatchNorm::BatchNorm(int num_features, float momentum, float eps,
                             const std::string& device)
            : num_features_(num_features), momentum_(momentum), eps_(eps),
              device_(device)
        {
            gamma_.resize(num_features_);
            beta_.resize(num_features_);
            gamma_.setConstant(1.0f);
            beta_.setZero();

            grad_gamma_.resize(num_features_);
            grad_beta_.resize(num_features_);
            grad_gamma_.setZero();
            grad_beta_.setZero();

            running_mean_.resize(num_features_);
            running_var_.resize(num_features_);
            running_mean_.setZero();
            running_var_.setConstant(1.0f);

            batch_mean_.resize(num_features_);
            batch_var_.resize(num_features_);
        }

        BatchNorm::~BatchNorm() = default;

        const Eigen::Tensor<float, 2> BatchNorm::forward(
            const Eigen::Tensor<float, 2>& input)
        {
            int batch = input.dimension(0);
            int features = input.dimension(1);

            if (features != num_features_)
                throw std::runtime_error("BatchNorm: input features (" +
                    std::to_string(features) + ") != num_features (" +
                    std::to_string(num_features_) + ")");

            input_cache_ = input;
            batch_size_cache_ = batch;

            Eigen::Tensor<float, 2> output(batch, features);
            x_hat_.resize(batch, features);

            if (training_)
            {
                // Compute batch mean
                batch_mean_.setZero();
                for (int n = 0; n < batch; ++n)
                    for (int f = 0; f < features; ++f)
                        batch_mean_(f) += input(n, f);
                float inv_batch = 1.0f / static_cast<float>(batch);
                for (int f = 0; f < features; ++f)
                    batch_mean_(f) *= inv_batch;

                // Compute batch variance
                batch_var_.setZero();
                for (int n = 0; n < batch; ++n)
                    for (int f = 0; f < features; ++f)
                    {
                        float diff = input(n, f) - batch_mean_(f);
                        batch_var_(f) += diff * diff;
                    }
                for (int f = 0; f < features; ++f)
                    batch_var_(f) *= inv_batch;

                // Normalize
                for (int n = 0; n < batch; ++n)
                    for (int f = 0; f < features; ++f)
                    {
                        x_hat_(n, f) = (input(n, f) - batch_mean_(f)) /
                                       std::sqrt(batch_var_(f) + eps_);
                        output(n, f) = gamma_(f) * x_hat_(n, f) + beta_(f);
                    }

                // Update running statistics (EMA)
                for (int f = 0; f < features; ++f)
                {
                    running_mean_(f) = (1.0f - momentum_) * running_mean_(f) +
                                       momentum_ * batch_mean_(f);
                    running_var_(f) = (1.0f - momentum_) * running_var_(f) +
                                      momentum_ * batch_var_(f);
                }
            }
            else
            {
                // Inference mode: use running statistics
                for (int n = 0; n < batch; ++n)
                    for (int f = 0; f < features; ++f)
                    {
                        x_hat_(n, f) = (input(n, f) - running_mean_(f)) /
                                       std::sqrt(running_var_(f) + eps_);
                        output(n, f) = gamma_(f) * x_hat_(n, f) + beta_(f);
                    }
            }

            return output;
        }

        const Eigen::Tensor<float, 2> BatchNorm::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            int batch = grad_output.dimension(0);
            int features = grad_output.dimension(1);
            float inv_batch = 1.0f / static_cast<float>(batch);

            grad_gamma_.setZero();
            grad_beta_.setZero();

            // grad_gamma = sum(grad_output * x_hat, axis=0)
            // grad_beta  = sum(grad_output, axis=0)
            for (int n = 0; n < batch; ++n)
                for (int f = 0; f < features; ++f)
                {
                    grad_gamma_(f) += grad_output(n, f) * x_hat_(n, f);
                    grad_beta_(f) += grad_output(n, f);
                }

            // Backprop through normalization
            // dx_hat = grad_output * gamma
            // dvar   = sum(dx_hat * (x - mean) * -0.5 * (var + eps)^{-3/2})
            // dmean  = sum(dx_hat * -1/sqrt(var+eps)) + dvar * mean(-2*(x-mean))/N
            // dx     = dx_hat / sqrt(var+eps) + dvar * 2*(x-mean)/N + dmean/N

            Eigen::Tensor<float, 2> grad_input(batch, features);

            for (int f = 0; f < features; ++f)
            {
                float inv_std = 1.0f / std::sqrt(batch_var_(f) + eps_);

                // dx_hat per sample, accumulate dvar and dmean
                float dvar = 0.0f;
                float dmean = 0.0f;

                for (int n = 0; n < batch; ++n)
                {
                    float dx_hat = grad_output(n, f) * gamma_(f);
                    float x_mu = input_cache_(n, f) - batch_mean_(f);

                    dvar += dx_hat * x_mu * (-0.5f) * inv_std * inv_std * inv_std;
                    dmean += dx_hat * (-inv_std);
                }

                // Second pass: compute grad_input
                for (int n = 0; n < batch; ++n)
                {
                    float dx_hat = grad_output(n, f) * gamma_(f);
                    float x_mu = input_cache_(n, f) - batch_mean_(f);

                    grad_input(n, f) = dx_hat * inv_std +
                                       dvar * 2.0f * x_mu * inv_batch +
                                       dmean * inv_batch;
                }
            }

            return grad_input;
        }

        void BatchNorm::step(Optimizers::Optimizer& /*optimizer*/, float learning_rate)
        {
            if (!trainable_) return;

            const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

            if (!adam_initialized_)
            {
                m_gamma_.resize(num_features_); m_gamma_.setZero();
                v_gamma_.resize(num_features_); v_gamma_.setZero();
                m_beta_.resize(num_features_);  m_beta_.setZero();
                v_beta_.resize(num_features_);  v_beta_.setZero();
                adam_initialized_ = true;
            }

            ++adam_t_;
            float bc1 = 1.0f - std::pow(beta1, static_cast<float>(adam_t_));
            float bc2 = 1.0f - std::pow(beta2, static_cast<float>(adam_t_));

            for (int f = 0; f < num_features_; ++f)
            {
                // Adam update for gamma
                m_gamma_(f) = beta1 * m_gamma_(f) + (1.0f - beta1) * grad_gamma_(f);
                v_gamma_(f) = beta2 * v_gamma_(f) + (1.0f - beta2) * grad_gamma_(f) * grad_gamma_(f);
                float mh_g = m_gamma_(f) / bc1;
                float vh_g = v_gamma_(f) / bc2;
                gamma_(f) -= learning_rate * mh_g / (std::sqrt(vh_g) + eps);

                // Adam update for beta
                m_beta_(f) = beta1 * m_beta_(f) + (1.0f - beta1) * grad_beta_(f);
                v_beta_(f) = beta2 * v_beta_(f) + (1.0f - beta2) * grad_beta_(f) * grad_beta_(f);
                float mh_b = m_beta_(f) / bc1;
                float vh_b = v_beta_(f) / bc2;
                beta_(f) -= learning_rate * mh_b / (std::sqrt(vh_b) + eps);
            }

            grad_gamma_.setZero();
            grad_beta_.setZero();
        }
    }
}
