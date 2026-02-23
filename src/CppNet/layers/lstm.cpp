/**
 * @file lstm.cpp
 * @brief LSTM layer implementation with full BPTT backward pass
 *
 * Forward:
 *   gates = x_t * W_ih + h_{t-1} * W_hh + bias      (4*hidden_size cols)
 *   i = sigmoid(gates[:, 0:H])
 *   f = sigmoid(gates[:, H:2H])
 *   g = tanh   (gates[:, 2H:3H])
 *   o = sigmoid(gates[:, 3H:4H])
 *   c_t = f ◦ c_{t-1} + i ◦ g
 *   h_t = o ◦ tanh(c_t)
 */

#include "CppNet/layers/lstm.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/utils/init.hpp"
#include <cmath>
#include <stdexcept>

namespace CppNet
{
    namespace Layers
    {
        // Helpers
        static inline float sigmoid(float x)
        {
            return 1.0f / (1.0f + std::exp(-x));
        }

        LSTM::LSTM(int input_size, int hidden_size, bool return_sequences,
                   const std::string& device)
            : input_size_(input_size), hidden_size_(hidden_size),
              return_sequences_(return_sequences), device_(device)
        {
            int gate4 = 4 * hidden_size_;

            // Xavier uniform with per-gate fan_out = hidden_size_
            float limit_ih = std::sqrt(6.0f / static_cast<float>(input_size_ + hidden_size_));
            float limit_hh = std::sqrt(6.0f / static_cast<float>(hidden_size_ + hidden_size_));

            W_ih_ = CppNet::Utils::uniform_init(input_size_, gate4, -limit_ih, limit_ih);
            W_hh_ = CppNet::Utils::uniform_init(hidden_size_, gate4, -limit_hh, limit_hh);
            bias_.resize(gate4);
            bias_.setZero();

            // Forget-gate bias trick: initialize to 1 so forget gate starts open
            for (int d = hidden_size_; d < 2 * hidden_size_; ++d)
                bias_(d) = 1.0f;

            grad_W_ih_.resize(input_size_, gate4);
            grad_W_hh_.resize(hidden_size_, gate4);
            grad_bias_.resize(gate4);
            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        LSTM::~LSTM() = default;

        const Eigen::Tensor<float, 3> LSTM::forward(
            const Eigen::Tensor<float, 3>& input)
        {
            // input: [batch, seq_len, input_size]
            input_cache_ = input;
            int batch = input.dimension(0);
            int seq_len = input.dimension(1);
            int gate4 = 4 * hidden_size_;

            // Initialize h0, c0
            h0_.resize(batch, hidden_size_);
            c0_.resize(batch, hidden_size_);
            h0_.setZero();
            c0_.setZero();

            caches_.clear();
            caches_.resize(seq_len);

            Eigen::Tensor<float, 3> output(batch, seq_len, hidden_size_);

            Eigen::Tensor<float, 2> h_prev = h0_;
            Eigen::Tensor<float, 2> c_prev = c0_;

            Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

            for (int t = 0; t < seq_len; ++t)
            {
                // Extract x_t [batch, input_size]
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input(n, t, d);

                // gates = x_t * W_ih + h_prev * W_hh + bias
                Eigen::Tensor<float, 2> gates = x_t.contract(W_ih_, contract_dims) +
                                                  h_prev.contract(W_hh_, contract_dims);

                // Add bias and apply activations
                Eigen::Tensor<float, 2> i_gate(batch, hidden_size_);
                Eigen::Tensor<float, 2> f_gate(batch, hidden_size_);
                Eigen::Tensor<float, 2> g_gate(batch, hidden_size_);
                Eigen::Tensor<float, 2> o_gate(batch, hidden_size_);
                Eigen::Tensor<float, 2> c_t(batch, hidden_size_);
                Eigen::Tensor<float, 2> h_t(batch, hidden_size_);
                Eigen::Tensor<float, 2> tanh_c(batch, hidden_size_);

                for (int n = 0; n < batch; ++n)
                {
                    for (int d = 0; d < hidden_size_; ++d)
                    {
                        i_gate(n, d) = sigmoid(gates(n, d + 0 * hidden_size_) +
                                               bias_(d + 0 * hidden_size_));
                        f_gate(n, d) = sigmoid(gates(n, d + 1 * hidden_size_) +
                                               bias_(d + 1 * hidden_size_));
                        g_gate(n, d) = std::tanh(gates(n, d + 2 * hidden_size_) +
                                                 bias_(d + 2 * hidden_size_));
                        o_gate(n, d) = sigmoid(gates(n, d + 3 * hidden_size_) +
                                               bias_(d + 3 * hidden_size_));

                        c_t(n, d) = f_gate(n, d) * c_prev(n, d) + i_gate(n, d) * g_gate(n, d);
                        tanh_c(n, d) = std::tanh(c_t(n, d));
                        h_t(n, d) = o_gate(n, d) * tanh_c(n, d);

                        output(n, t, d) = h_t(n, d);
                    }
                }

                // Cache for backward
                caches_[t].i_gate = i_gate;
                caches_[t].f_gate = f_gate;
                caches_[t].g_gate = g_gate;
                caches_[t].o_gate = o_gate;
                caches_[t].c_t = c_t;
                caches_[t].h_t = h_t;
                caches_[t].tanh_c = tanh_c;

                h_prev = h_t;
                c_prev = c_t;
            }

            return output;
        }

        const Eigen::Tensor<float, 3> LSTM::backward(
            const Eigen::Tensor<float, 3>& grad_output)
        {
            // grad_output: [batch, seq_len, hidden_size]
            int batch = grad_output.dimension(0);
            int seq_len = grad_output.dimension(1);
            int gate4 = 4 * hidden_size_;

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();

            Eigen::Tensor<float, 3> grad_input(batch, seq_len, input_size_);
            grad_input.setZero();

            Eigen::Tensor<float, 2> dh_next(batch, hidden_size_);
            Eigen::Tensor<float, 2> dc_next(batch, hidden_size_);
            dh_next.setZero();
            dc_next.setZero();

            for (int t = seq_len - 1; t >= 0; --t)
            {
                auto& cache = caches_[t];

                // dh = grad_output[:, t, :] + dh_next
                Eigen::Tensor<float, 2> dh(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dh(n, d) = grad_output(n, t, d) + dh_next(n, d);

                // dc = dh * o * (1 - tanh_c^2) + dc_next
                Eigen::Tensor<float, 2> dc(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dc(n, d) = dh(n, d) * cache.o_gate(n, d) *
                                   (1.0f - cache.tanh_c(n, d) * cache.tanh_c(n, d)) +
                                   dc_next(n, d);

                // Gate gradients (pre-activation)
                Eigen::Tensor<float, 2> dgates(batch, gate4);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                    {
                        // di = dc * g * i * (1 - i)
                        float di = dc(n, d) * cache.g_gate(n, d) *
                                   cache.i_gate(n, d) * (1.0f - cache.i_gate(n, d));
                        // df = dc * c_{t-1} * f * (1 - f)
                        float c_prev_val = (t > 0) ? caches_[t - 1].c_t(n, d) : 0.0f;
                        float df = dc(n, d) * c_prev_val *
                                   cache.f_gate(n, d) * (1.0f - cache.f_gate(n, d));
                        // dg = dc * i * (1 - g^2)
                        float dg = dc(n, d) * cache.i_gate(n, d) *
                                   (1.0f - cache.g_gate(n, d) * cache.g_gate(n, d));
                        // do = dh * tanh_c * o * (1 - o)
                        float do_ = dh(n, d) * cache.tanh_c(n, d) *
                                    cache.o_gate(n, d) * (1.0f - cache.o_gate(n, d));

                        dgates(n, d + 0 * hidden_size_) = di;
                        dgates(n, d + 1 * hidden_size_) = df;
                        dgates(n, d + 2 * hidden_size_) = dg;
                        dgates(n, d + 3 * hidden_size_) = do_;
                    }

                // Extract x_t and h_prev
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input_cache_(n, t, d);

                Eigen::Tensor<float, 2> h_prev = (t > 0) ? caches_[t - 1].h_t : h0_;

                // Accumulate weight gradients
                Eigen::array<Eigen::IndexPair<int>, 1> contract_batch = {Eigen::IndexPair<int>(0, 0)};
                grad_W_ih_ += x_t.contract(dgates, contract_batch);      // [input, 4H]
                grad_W_hh_ += h_prev.contract(dgates, contract_batch);   // [hidden, 4H]

                // Bias gradient
                for (int g = 0; g < gate4; ++g)
                    for (int n = 0; n < batch; ++n)
                        grad_bias_(g) += dgates(n, g);

                // grad_input[:, t, :] = dgates * W_ih^T
                Eigen::array<Eigen::IndexPair<int>, 1> contract_gate = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> dx = dgates.contract(W_ih_, contract_gate);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        grad_input(n, t, d) = dx(n, d);

                // dh_next = dgates * W_hh^T
                dh_next = dgates.contract(W_hh_, contract_gate);

                // dc_next = dc * f_gate
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dc_next(n, d) = dc(n, d) * cache.f_gate(n, d);
            }

            return grad_input;
        }

        void LSTM::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            if (!trainable_) return;

            optimizer.update(W_ih_.data(), grad_W_ih_.data(), W_ih_.size(), learning_rate);
            optimizer.update(W_hh_.data(), grad_W_hh_.data(), W_hh_.size(), learning_rate);
            optimizer.update(bias_.data(), grad_bias_.data(), bias_.size(), learning_rate);

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }
    }
}
