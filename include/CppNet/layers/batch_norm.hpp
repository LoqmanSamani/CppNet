/**
 * @file batch_norm.hpp
 * @brief Batch Normalization layer
 *
 * Normalizes each feature across the batch dimension, then applies
 * a learnable affine transform: y = gamma * x_hat + beta.
 *
 * During training: uses batch statistics and updates running stats.
 * During inference: uses running (exponential moving average) statistics.
 *
 * Reference: Ioffe & Szegedy, "Batch Normalization: Accelerating Deep
 * Network Training by Reducing Internal Covariate Shift", 2015.
 */

#ifndef BATCH_NORM_HPP
#define BATCH_NORM_HPP

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class BatchNorm
         * @brief 1D Batch Normalization over features
         *
         * Input:  [batch, num_features]
         * Output: [batch, num_features]
         */
        class BatchNorm : public Layer
        {
        public:
            /**
             * @param num_features Number of features (channels)
             * @param momentum     EMA momentum for running stats (default 0.1)
             * @param eps          Small constant for numerical stability (default 1e-5)
             * @param device       Compute backend
             */
            BatchNorm(int num_features, float momentum = 0.1f, float eps = 1e-5f,
                      const std::string& device = "cpu-eigen");
            ~BatchNorm();

            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input);
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output);

            bool is_trainable() const override { return trainable_; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

            void train() { training_ = true; }
            void eval() { training_ = false; }

            void freeze() { trainable_ = false; }
            void unfreeze() { trainable_ = true; }

            int get_num_features() const { return num_features_; }

            Eigen::Tensor<float, 1>& get_gamma() { return gamma_; }
            Eigen::Tensor<float, 1>& get_beta() { return beta_; }
            const Eigen::Tensor<float, 1>& get_gamma() const { return gamma_; }
            const Eigen::Tensor<float, 1>& get_beta() const { return beta_; }
            const Eigen::Tensor<float, 1>& get_grad_gamma() const { return grad_gamma_; }
            const Eigen::Tensor<float, 1>& get_grad_beta() const { return grad_beta_; }

            const Eigen::Tensor<float, 1>& get_running_mean() const { return running_mean_; }
            const Eigen::Tensor<float, 1>& get_running_var() const { return running_var_; }

            void set_running_mean(const Eigen::Tensor<float, 1>& m) { running_mean_ = m; }
            void set_running_var(const Eigen::Tensor<float, 1>& v) { running_var_ = v; }

        private:
            int num_features_;
            float momentum_;
            float eps_;
            bool training_ = true;
            bool trainable_ = true;
            std::string device_;

            // Learnable parameters
            Eigen::Tensor<float, 1> gamma_;   // [num_features]  scale
            Eigen::Tensor<float, 1> beta_;    // [num_features]  shift

            // Gradients
            Eigen::Tensor<float, 1> grad_gamma_;
            Eigen::Tensor<float, 1> grad_beta_;

            // Running statistics (for inference)
            Eigen::Tensor<float, 1> running_mean_;
            Eigen::Tensor<float, 1> running_var_;

            // Cached values for backward
            Eigen::Tensor<float, 2> x_hat_;       // normalized input
            Eigen::Tensor<float, 1> batch_mean_;
            Eigen::Tensor<float, 1> batch_var_;
            Eigen::Tensor<float, 2> input_cache_;
            int batch_size_cache_ = 0;
        };
    }
}

#endif // BATCH_NORM_HPP