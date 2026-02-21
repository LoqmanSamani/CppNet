/**
 * @file gru.hpp
 * @brief Gated Recurrent Unit (GRU) layer
 *
 * Gate equations:
 *   z_t = sigmoid(x_t * W_iz + h_{t-1} * W_hz + b_z)       update gate
 *   r_t = sigmoid(x_t * W_ir + h_{t-1} * W_hr + b_r)       reset gate
 *   n_t = tanh   (x_t * W_in + (r_t ◦ h_{t-1}) * W_hn + b_n)  candidate
 *   h_t = (1 - z_t) ◦ n_t + z_t ◦ h_{t-1}
 *
 * Weights stored concatenated:
 *   W_ih: [input_size,  3 * hidden_size]   (iz | ir | in)
 *   W_hh: [hidden_size, 3 * hidden_size]   (hz | hr | hn)
 *   bias: [3 * hidden_size]
 */

#ifndef GRU_HPP
#define GRU_HPP

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
         * @class GRU
         * @brief Single-layer GRU
         *
         * Input:  [batch, seq_len, input_size]
         * Output: [batch, seq_len, hidden_size]
         */
        class GRU : public Layer
        {
        public:
            GRU(int input_size, int hidden_size, bool return_sequences = true,
                const std::string& device = "cpu-eigen");
            ~GRU();

            const Eigen::Tensor<float, 3> forward(const Eigen::Tensor<float, 3>& input);
            const Eigen::Tensor<float, 3> backward(const Eigen::Tensor<float, 3>& grad_output);

            bool is_trainable() const override { return trainable_; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

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

            Eigen::Tensor<float, 2> W_ih_;   // [input_size,  3*hidden_size]
            Eigen::Tensor<float, 2> W_hh_;   // [hidden_size, 3*hidden_size]
            Eigen::Tensor<float, 1> bias_;   // [3*hidden_size]

            Eigen::Tensor<float, 2> grad_W_ih_;
            Eigen::Tensor<float, 2> grad_W_hh_;
            Eigen::Tensor<float, 1> grad_bias_;

            struct TimeCache
            {
                Eigen::Tensor<float, 2> z_gate;   // update gate [batch, hidden]
                Eigen::Tensor<float, 2> r_gate;   // reset gate
                Eigen::Tensor<float, 2> n_cand;   // candidate hidden
                Eigen::Tensor<float, 2> h_t;      // hidden state
                Eigen::Tensor<float, 2> h_prev;   // previous hidden
                Eigen::Tensor<float, 2> rh;       // r_t ◦ h_{t-1}
            };

            std::vector<TimeCache> caches_;
            Eigen::Tensor<float, 3> input_cache_;
            Eigen::Tensor<float, 2> h0_;

            // Adam optimizer state
            Eigen::Tensor<float, 2> m_ih_, v_ih_, m_hh_, v_hh_;
            Eigen::Tensor<float, 1> m_b_, v_b_;
            int adam_t_ = 0;
            bool adam_initialized_ = false;
        };
    }
}

#endif // GRU_HPP
