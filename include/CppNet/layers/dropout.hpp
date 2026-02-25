/**
 * @file dropout.hpp
 * @brief Dropout regularization layer
 *
 * During training, randomly zeroes elements with probability p
 * and scales remaining elements by 1/(1-p) (inverted dropout).
 * During inference, acts as identity.
 */

#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <random>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class Dropout
         * @brief Inverted dropout layer
         *
         * 2D input:  [batch, features]
         * 4D input:  [batch, channels, height, width]
         */
        class Dropout : public Layer
        {
        public:
            /**
             * @param p Probability of an element being zeroed (default 0.5)
             */
            explicit Dropout(float p = 0.5f);
            ~Dropout();

            /// Forward pass for 2D tensors [batch, features]
            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input);

            /// Forward pass for 4D tensors [batch, channels, H, W]
            Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& input);

            /// Backward pass for 2D tensors
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output);

            /// Backward pass for 4D tensors
            Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output);

            bool is_trainable() const override { return false; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

            void train() { training_ = true; }
            void eval() { training_ = false; }
            bool is_training() const { return training_; }
            float get_p() const { return p_; }

        private:
            float p_;           // drop probability
            float scale_;       // 1 / (1 - p)
            bool training_ = true;

            // Masks cached for backward
            Eigen::Tensor<float, 2> mask_2d_;
            Eigen::Tensor<float, 4> mask_4d_;

            std::mt19937 gen_;
            std::bernoulli_distribution dist_;
        };
    }
}

#endif // DROPOUT_HPP
