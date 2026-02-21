/**
 * @file rnn.cpp
 * @brief Vanilla RNN layer implementation
 *
 * h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
 * Backward: BPTT (backpropagation through time)
 */

#include "CppNet/layers/rnn.hpp"
#include <cmath>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Layers
    {
        RNN::RNN(int input_size, int hidden_size, bool return_sequences,
                 const std::string& device)
            : input_size_(input_size), hidden_size_(hidden_size),
              return_sequences_(return_sequences), device_(device)
        {
            // Xavier initialization
            float limit_ih = std::sqrt(6.0f / static_cast<float>(input_size_ + hidden_size_));
            float limit_hh = std::sqrt(6.0f / static_cast<float>(hidden_size_ + hidden_size_));

            W_ih_.resize(input_size_, hidden_size_);
            W_hh_.resize(hidden_size_, hidden_size_);
            bias_.resize(hidden_size_);

            W_ih_.setRandom();
            W_ih_ = W_ih_ * W_ih_.constant(limit_ih);
            W_hh_.setRandom();
            W_hh_ = W_hh_ * W_hh_.constant(limit_hh);
            bias_.setZero();

            grad_W_ih_.resize(input_size_, hidden_size_);
            grad_W_hh_.resize(hidden_size_, hidden_size_);
            grad_bias_.resize(hidden_size_);
            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        RNN::~RNN() = default;

        const Eigen::Tensor<float, 3> RNN::forward(const Eigen::Tensor<float, 3>& input)
        {
            // input: [batch, seq_len, input_size]
            input_cache_ = input;
            int batch = input.dimension(0);
            int seq_len = input.dimension(1);

            hidden_states_.clear();
            hidden_states_.reserve(seq_len + 1);

            // h_0 = zeros
            Eigen::Tensor<float, 2> h_prev(batch, hidden_size_);
            h_prev.setZero();
            hidden_states_.push_back(h_prev);

            Eigen::Tensor<float, 3> output(batch, seq_len, hidden_size_);

            Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

            for (int t = 0; t < seq_len; ++t)
            {
                // Extract x_t: [batch, input_size]
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input(n, t, d);

                // h_t = tanh(x_t * W_ih + h_{t-1} * W_hh + b)
                Eigen::Tensor<float, 2> pre_act = x_t.contract(W_ih_, contract_dims) +
                                                   h_prev.contract(W_hh_, contract_dims);

                Eigen::Tensor<float, 2> h_t(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                    {
                        h_t(n, d) = std::tanh(pre_act(n, d) + bias_(d));
                        output(n, t, d) = h_t(n, d);
                    }

                hidden_states_.push_back(h_t);
                h_prev = h_t;
            }

            return output;
        }

        const Eigen::Tensor<float, 3> RNN::backward(const Eigen::Tensor<float, 3>& grad_output)
        {
            // grad_output: [batch, seq_len, hidden_size]
            int batch = grad_output.dimension(0);
            int seq_len = grad_output.dimension(1);

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();

            Eigen::Tensor<float, 3> grad_input(batch, seq_len, input_size_);
            grad_input.setZero();

            Eigen::Tensor<float, 2> dh_next(batch, hidden_size_);
            dh_next.setZero();

            for (int t = seq_len - 1; t >= 0; --t)
            {
                // dh = grad_output[:, t, :] + dh_next
                Eigen::Tensor<float, 2> dh(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dh(n, d) = grad_output(n, t, d) + dh_next(n, d);

                // dtanh = dh * (1 - h_t^2)
                auto& h_t = hidden_states_[t + 1];
                Eigen::Tensor<float, 2> dtanh(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dtanh(n, d) = dh(n, d) * (1.0f - h_t(n, d) * h_t(n, d));

                // Extract x_t and h_{t-1}
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input_cache_(n, t, d);

                auto& h_prev = hidden_states_[t];

                // Accumulate gradients
                // grad_W_ih += x_t^T * dtanh
                Eigen::array<Eigen::IndexPair<int>, 1> contract_batch = {Eigen::IndexPair<int>(0, 0)};
                grad_W_ih_ += x_t.contract(dtanh, contract_batch);
                grad_W_hh_ += h_prev.contract(dtanh, contract_batch);

                // grad_bias += sum(dtanh, axis=0)
                for (int d = 0; d < hidden_size_; ++d)
                    for (int n = 0; n < batch; ++n)
                        grad_bias_(d) += dtanh(n, d);

                // grad_input[:, t, :] = dtanh * W_ih^T
                Eigen::array<Eigen::IndexPair<int>, 1> contract_hidden = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> dx = dtanh.contract(W_ih_, contract_hidden);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        grad_input(n, t, d) = dx(n, d);

                // dh_next = dtanh * W_hh^T
                dh_next = dtanh.contract(W_hh_, contract_hidden);
            }

            return grad_input;
        }

        void RNN::step(Optimizers::Optimizer& /*optimizer*/, float /*learning_rate*/)
        {
            // TODO: Integrate with optimizer once it supports RNN
        }
    }
}
