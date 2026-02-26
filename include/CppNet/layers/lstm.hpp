/**
 * @file lstm.hpp
 * @brief Long Short-Term Memory (LSTM) layer
 *
 * Gate equations:
 *   i_t = sigmoid(x_t * W_ii + h_{t-1} * W_hi + b_i)       input gate
 *   f_t = sigmoid(x_t * W_if + h_{t-1} * W_hf + b_f)       forget gate
 *   g_t = tanh   (x_t * W_ig + h_{t-1} * W_hg + b_g)       cell candidate
 *   o_t = sigmoid(x_t * W_io + h_{t-1} * W_ho + b_o)       output gate
 *   c_t = f_t * c_{t-1} + i_t * g_t
 *   h_t = o_t * tanh(c_t)
 *
 * Weights are stored as concatenated matrices for efficiency:
 *   W_ih: [input_size,  4 * hidden_size]   (ii | if | ig | io)
 *   W_hh: [hidden_size, 4 * hidden_size]   (hi | hf | hg | ho)
 *   bias: [4 * hidden_size]
 */

#ifndef LSTM_HPP
#define LSTM_HPP

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
         * @class LSTM
         * @brief Single-layer LSTM
         *
         * Input:  [batch, seq_len, input_size]
         * Output: [batch, seq_len, hidden_size]   (return_sequences=true)
         *    or   [batch, hidden_size]             (return_sequences=false, last h only)
         */
        class LSTM : public Layer
        {
        public:
            /**
             * @param input_size       Dimension of input features
             * @param hidden_size      Dimension of hidden / cell state
             * @param return_sequences If true, return all timesteps
             * @param device           Compute backend
             */
            LSTM(int input_size, int hidden_size, bool return_sequences = true,
                 const std::string& device = "cpu-eigen");
            ~LSTM();

            const Eigen::Tensor<float, 3> forward(const Eigen::Tensor<float, 3>& input);
            const Eigen::Tensor<float, 3> backward(const Eigen::Tensor<float, 3>& grad_output);

            bool is_trainable() const override { return trainable_; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;
            void reset_grads() override;

            void freeze() { trainable_ = false; }
            void unfreeze() { trainable_ = true; }

            int get_input_size() const { return input_size_; }
            int get_hidden_size() const { return hidden_size_; }

            Eigen::Tensor<float, 2>& get_W_ih() { return W_ih_; }
            Eigen::Tensor<float, 2>& get_W_hh() { return W_hh_; }
            Eigen::Tensor<float, 1>& get_bias() { return bias_; }

        private:
            int input_size_;
            int hidden_size_;
            bool return_sequences_;
            bool trainable_ = true;
            std::string device_;

            // Concatenated weight matrices  [... , 4*hidden_size]
            Eigen::Tensor<float, 2> W_ih_;   // [input_size,  4*hidden_size]
            Eigen::Tensor<float, 2> W_hh_;   // [hidden_size, 4*hidden_size]
            Eigen::Tensor<float, 1> bias_;   // [4*hidden_size]

            Eigen::Tensor<float, 2> grad_W_ih_;
            Eigen::Tensor<float, 2> grad_W_hh_;
            Eigen::Tensor<float, 1> grad_bias_;

            // Per-timestep caches for BPTT
            struct TimeCache
            {
                Eigen::Tensor<float, 2> i_gate;  // [batch, hidden]
                Eigen::Tensor<float, 2> f_gate;
                Eigen::Tensor<float, 2> g_gate;
                Eigen::Tensor<float, 2> o_gate;
                Eigen::Tensor<float, 2> c_t;     // cell state
                Eigen::Tensor<float, 2> h_t;     // hidden state
                Eigen::Tensor<float, 2> tanh_c;  // tanh(c_t)
            };

            std::vector<TimeCache> caches_;
            Eigen::Tensor<float, 3> input_cache_;

            // Initial states (zero)
            Eigen::Tensor<float, 2> h0_;
            Eigen::Tensor<float, 2> c0_;

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

#endif // LSTM_HPP