/**
 * @file rnn.hpp
 * @brief Vanilla Recurrent Neural Network (RNN) layer
 */

#ifndef RNN_HPP
#define RNN_HPP

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class RNN
         * @brief Vanilla RNN layer: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
         *
         * Input:  [batch, seq_len, input_size]
         * Output: [batch, seq_len, hidden_size] (all timesteps)
         *    or   [batch, hidden_size] (last timestep only, if return_sequences=false)
         */
        class RNN : public Layer
        {
        public:
            /**
             * @param input_size Dimension of input features
             * @param hidden_size Dimension of hidden state
             * @param return_sequences If true, return output at all timesteps
             * @param device Compute backend
             */
            RNN(int input_size, int hidden_size, bool return_sequences = true,
                const std::string& device = "cpu-eigen");
            ~RNN();

            const Eigen::Tensor<float, 3> forward(const Eigen::Tensor<float, 3>& input);
            const Eigen::Tensor<float, 3> backward(const Eigen::Tensor<float, 3>& grad_output);

            bool is_trainable() const override { return trainable_; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;
            void reset_grads() override;

            void freeze() { trainable_ = false; }
            void unfreeze() { trainable_ = true; }

            int get_input_size() const { return input_size_; }
            int get_hidden_size() const { return hidden_size_; }

        private:
            int input_size_;
            int hidden_size_;
            bool return_sequences_;
            bool trainable_ = true;
            std::string device_;

            Eigen::Tensor<float, 2> W_ih_;  // [input_size, hidden_size]
            Eigen::Tensor<float, 2> W_hh_;  // [hidden_size, hidden_size]
            Eigen::Tensor<float, 1> bias_;  // [hidden_size]

            Eigen::Tensor<float, 2> grad_W_ih_;
            Eigen::Tensor<float, 2> grad_W_hh_;
            Eigen::Tensor<float, 1> grad_bias_;

            // Caches for backward pass
            std::vector<Eigen::Tensor<float, 2>> hidden_states_;  // per timestep
            Eigen::Tensor<float, 3> input_cache_;

            #ifdef USE_CUDA
            void forward_gpu(const Eigen::Tensor<float, 3>& input,
                             Eigen::Tensor<float, 3>& output,
                             int batch, int seq_len);
            void backward_gpu(const Eigen::Tensor<float, 3>& grad_output,
                              Eigen::Tensor<float, 3>& grad_input,
                              int batch, int seq_len);
            #endif
        };
    }
}

#endif // RNN_HPP
