/**
 * @file gru.cpp
 * @brief GRU layer implementation with BPTT backward pass
 *
 * Forward (per timestep):
 *   z_t = sigmoid(x_t * W_iz + h_{t-1} * W_hz + b_z)        update gate
 *   r_t = sigmoid(x_t * W_ir + h_{t-1} * W_hr + b_r)        reset gate
 *   n_t = tanh   (x_t * W_in + (r_t ◦ h_{t-1}) * W_hn + b_n)  candidate
 *   h_t = (1 - z_t) ◦ n_t + z_t ◦ h_{t-1}
 */

#include "CppNet/layers/gru.hpp"
#include <cmath>

namespace CppNet
{
    namespace Layers
    {
        static inline float sigmoid(float x)
        {
            return 1.0f / (1.0f + std::exp(-x));
        }

        GRU::GRU(int input_size, int hidden_size, bool return_sequences,
                 const std::string& device)
            : input_size_(input_size), hidden_size_(hidden_size),
              return_sequences_(return_sequences), device_(device)
        {
            int gate3 = 3 * hidden_size_;

            float limit_ih = std::sqrt(6.0f / static_cast<float>(input_size_ + hidden_size_));
            float limit_hh = std::sqrt(6.0f / static_cast<float>(hidden_size_ + hidden_size_));

            W_ih_.resize(input_size_, gate3);
            W_hh_.resize(hidden_size_, gate3);
            bias_.resize(gate3);

            W_ih_.setRandom();
            W_ih_ = W_ih_ * W_ih_.constant(limit_ih);
            W_hh_.setRandom();
            W_hh_ = W_hh_ * W_hh_.constant(limit_hh);
            bias_.setZero();

            grad_W_ih_.resize(input_size_, gate3);
            grad_W_hh_.resize(hidden_size_, gate3);
            grad_bias_.resize(gate3);
            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        GRU::~GRU() = default;

        const Eigen::Tensor<float, 3> GRU::forward(
            const Eigen::Tensor<float, 3>& input)
        {
            input_cache_ = input;
            int batch = input.dimension(0);
            int seq_len = input.dimension(1);
            int H = hidden_size_;

            h0_.resize(batch, H);
            h0_.setZero();

            caches_.clear();
            caches_.resize(seq_len);

            Eigen::Tensor<float, 3> output(batch, seq_len, H);
            Eigen::Tensor<float, 2> h_prev = h0_;

            Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

            for (int t = 0; t < seq_len; ++t)
            {
                // Extract x_t [batch, input_size]
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input(n, t, d);

                // Compute all gates from concatenated weights
                // x_gates = x_t * W_ih  → [batch, 3*H]
                // h_gates = h_prev * W_hh → [batch, 3*H]
                Eigen::Tensor<float, 2> x_gates = x_t.contract(W_ih_, contract_dims);
                Eigen::Tensor<float, 2> h_gates = h_prev.contract(W_hh_, contract_dims);

                Eigen::Tensor<float, 2> z_gate(batch, H);
                Eigen::Tensor<float, 2> r_gate(batch, H);
                Eigen::Tensor<float, 2> n_cand(batch, H);
                Eigen::Tensor<float, 2> rh(batch, H);
                Eigen::Tensor<float, 2> h_t(batch, H);

                for (int n = 0; n < batch; ++n)
                {
                    for (int d = 0; d < H; ++d)
                    {
                        // z = sigmoid(x_gates_z + h_gates_z + b_z)
                        z_gate(n, d) = sigmoid(
                            x_gates(n, d + 0 * H) + h_gates(n, d + 0 * H) +
                            bias_(d + 0 * H));

                        // r = sigmoid(x_gates_r + h_gates_r + b_r)
                        r_gate(n, d) = sigmoid(
                            x_gates(n, d + 1 * H) + h_gates(n, d + 1 * H) +
                            bias_(d + 1 * H));

                        // rh = r ◦ h_prev
                        rh(n, d) = r_gate(n, d) * h_prev(n, d);
                    }
                }

                // n_cand needs rh * W_hn (the third hidden-state block)
                // We already have h_gates which contains h_prev * W_hh for all 3 blocks
                // But for the candidate we need (r ◦ h_prev) * W_hn, not h_prev * W_hn
                // So we recompute just the n-block: rh * W_hh[:, 2H:3H]
                // Extract W_hn sub-matrix
                Eigen::Tensor<float, 2> W_hn(hidden_size_, H);
                for (int i = 0; i < hidden_size_; ++i)
                    for (int j = 0; j < H; ++j)
                        W_hn(i, j) = W_hh_(i, j + 2 * H);

                Eigen::Tensor<float, 2> rh_proj = rh.contract(W_hn, contract_dims);

                for (int n = 0; n < batch; ++n)
                {
                    for (int d = 0; d < H; ++d)
                    {
                        // n = tanh(x_gates_n + rh_proj + b_n)
                        n_cand(n, d) = std::tanh(
                            x_gates(n, d + 2 * H) + rh_proj(n, d) +
                            bias_(d + 2 * H));

                        // h_t = (1 - z) * n + z * h_prev
                        h_t(n, d) = (1.0f - z_gate(n, d)) * n_cand(n, d) +
                                    z_gate(n, d) * h_prev(n, d);

                        output(n, t, d) = h_t(n, d);
                    }
                }

                caches_[t].z_gate = z_gate;
                caches_[t].r_gate = r_gate;
                caches_[t].n_cand = n_cand;
                caches_[t].h_t = h_t;
                caches_[t].h_prev = h_prev;
                caches_[t].rh = rh;

                h_prev = h_t;
            }

            return output;
        }

        const Eigen::Tensor<float, 3> GRU::backward(
            const Eigen::Tensor<float, 3>& grad_output)
        {
            int batch = grad_output.dimension(0);
            int seq_len = grad_output.dimension(1);
            int H = hidden_size_;
            int gate3 = 3 * H;

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();

            Eigen::Tensor<float, 3> grad_input(batch, seq_len, input_size_);
            grad_input.setZero();

            Eigen::Tensor<float, 2> dh_next(batch, H);
            dh_next.setZero();

            // Extract W_hn sub-matrix for candidate computation
            Eigen::Tensor<float, 2> W_hn(hidden_size_, H);
            for (int i = 0; i < hidden_size_; ++i)
                for (int j = 0; j < H; ++j)
                    W_hn(i, j) = W_hh_(i, j + 2 * H);

            for (int t = seq_len - 1; t >= 0; --t)
            {
                auto& cache = caches_[t];

                Eigen::Tensor<float, 2> dh(batch, H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        dh(n, d) = grad_output(n, t, d) + dh_next(n, d);

                // Backprop through h_t = (1-z)*n + z*h_prev
                Eigen::Tensor<float, 2> dn(batch, H);
                Eigen::Tensor<float, 2> dz(batch, H);
                Eigen::Tensor<float, 2> dh_prev(batch, H);

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                    {
                        dn(n, d) = dh(n, d) * (1.0f - cache.z_gate(n, d));
                        dz(n, d) = dh(n, d) * (cache.h_prev(n, d) - cache.n_cand(n, d));
                        dh_prev(n, d) = dh(n, d) * cache.z_gate(n, d);
                    }

                // dn through tanh: dn_raw = dn * (1 - n^2)
                Eigen::Tensor<float, 2> dn_raw(batch, H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        dn_raw(n, d) = dn(n, d) *
                            (1.0f - cache.n_cand(n, d) * cache.n_cand(n, d));

                // dz through sigmoid: dz_raw = dz * z * (1-z)
                Eigen::Tensor<float, 2> dz_raw(batch, H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        dz_raw(n, d) = dz(n, d) *
                            cache.z_gate(n, d) * (1.0f - cache.z_gate(n, d));

                // Backprop through r gate
                // dn_raw propagates through rh_proj = rh * W_hn
                // d_rh = dn_raw * W_hn^T
                Eigen::array<Eigen::IndexPair<int>, 1> contract_t = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> d_rh = dn_raw.contract(W_hn, contract_t);

                // d_r = d_rh * h_prev, dr through sigmoid
                Eigen::Tensor<float, 2> dr_raw(batch, H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                    {
                        float dr = d_rh(n, d) * cache.h_prev(n, d);
                        dr_raw(n, d) = dr * cache.r_gate(n, d) *
                                       (1.0f - cache.r_gate(n, d));
                        // Also add d_rh * r to dh_prev
                        dh_prev(n, d) += d_rh(n, d) * cache.r_gate(n, d);
                    }

                // Assemble dgates = [dz_raw | dr_raw | dn_raw]  [batch, 3*H]
                Eigen::Tensor<float, 2> dgates(batch, gate3);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                    {
                        dgates(n, d + 0 * H) = dz_raw(n, d);
                        dgates(n, d + 1 * H) = dr_raw(n, d);
                        dgates(n, d + 2 * H) = dn_raw(n, d);
                    }

                // x_t
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input_cache_(n, t, d);

                // W_ih gradient: x_t^T * dgates
                Eigen::array<Eigen::IndexPair<int>, 1> contract_batch = {Eigen::IndexPair<int>(0, 0)};
                grad_W_ih_ += x_t.contract(dgates, contract_batch);

                // W_hh gradient: h_prev^T * dgates
                // But for the n-block we need rh, not h_prev
                // Build h_for_grad: [h_prev for z&r blocks, rh for n block]
                Eigen::Tensor<float, 2> h_for_grad(batch, gate3);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                    {
                        h_for_grad(n, d + 0 * H) = cache.h_prev(n, d);
                        h_for_grad(n, d + 1 * H) = cache.h_prev(n, d);
                        h_for_grad(n, d + 2 * H) = cache.rh(n, d);
                    }
                // Actually we want: sum over batch of h_for_grad^T * dgates (element-wise per gate)
                // This is trickier. Let's just accumulate separately per gate group.
                // For z and r columns: h_prev^T * dgates[:, z_or_r]
                // For n column: rh^T * dgates[:, n]
                for (int n = 0; n < batch; ++n)
                {
                    for (int i = 0; i < hidden_size_; ++i)
                    {
                        for (int d = 0; d < H; ++d)
                        {
                            // z-gate
                            grad_W_hh_(i, d + 0 * H) += cache.h_prev(n, i) * dgates(n, d + 0 * H);
                            // r-gate
                            grad_W_hh_(i, d + 1 * H) += cache.h_prev(n, i) * dgates(n, d + 1 * H);
                            // n-candidate (uses rh)
                            grad_W_hh_(i, d + 2 * H) += cache.rh(n, i) * dgates(n, d + 2 * H);
                        }
                    }
                }

                // Bias gradient
                for (int g = 0; g < gate3; ++g)
                    for (int n = 0; n < batch; ++n)
                        grad_bias_(g) += dgates(n, g);

                // grad_input = dgates * W_ih^T
                Eigen::Tensor<float, 2> dx = dgates.contract(W_ih_, contract_t);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        grad_input(n, t, d) = dx(n, d);

                // dh_next = dh_prev + dgates[:, z&r] * W_hh[:, z&r]^T
                // The z&r contribution to h_prev through the gate projections
                Eigen::Tensor<float, 2> dgates_zr(batch, 2 * H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < 2 * H; ++d)
                        dgates_zr(n, d) = dgates(n, d);

                Eigen::Tensor<float, 2> W_hh_zr(hidden_size_, 2 * H);
                for (int i = 0; i < hidden_size_; ++i)
                    for (int j = 0; j < 2 * H; ++j)
                        W_hh_zr(i, j) = W_hh_(i, j);

                Eigen::array<Eigen::IndexPair<int>, 1> contract_2h = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> dh_from_gates = dgates_zr.contract(W_hh_zr, contract_2h);

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        dh_next(n, d) = dh_prev(n, d) + dh_from_gates(n, d);
            }

            return grad_input;
        }

        void GRU::step(Optimizers::Optimizer& /*optimizer*/, float learning_rate)
        {
            if (!trainable_) return;

            const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

            if (!adam_initialized_) {
                m_ih_.resize(W_ih_.dimensions()); m_ih_.setZero();
                v_ih_.resize(W_ih_.dimensions()); v_ih_.setZero();
                m_hh_.resize(W_hh_.dimensions()); m_hh_.setZero();
                v_hh_.resize(W_hh_.dimensions()); v_hh_.setZero();
                m_b_.resize(bias_.dimensions()); m_b_.setZero();
                v_b_.resize(bias_.dimensions()); v_b_.setZero();
                adam_initialized_ = true;
            }

            ++adam_t_;
            float bc1 = 1.0f - std::pow(beta1, adam_t_);
            float bc2 = 1.0f - std::pow(beta2, adam_t_);

            auto adam_update = [&](float* w, float* g, float* m, float* v, int n) {
                for (int i = 0; i < n; ++i) {
                    m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
                    v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
                    float mh = m[i] / bc1, vh = v[i] / bc2;
                    w[i] -= learning_rate * mh / (std::sqrt(vh) + eps);
                }
            };

            adam_update(W_ih_.data(), grad_W_ih_.data(), m_ih_.data(), v_ih_.data(), W_ih_.size());
            adam_update(W_hh_.data(), grad_W_hh_.data(), m_hh_.data(), v_hh_.data(), W_hh_.size());
            adam_update(bias_.data(), grad_bias_.data(), m_b_.data(), v_b_.data(), bias_.size());

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }
    }
}
