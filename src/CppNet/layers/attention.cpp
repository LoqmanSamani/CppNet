/**
 * @file attention.cpp
 * @brief Multi-Head Attention layer implementation
 *
 * Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 * Multi-head: split embed_dim into num_heads, apply attention per head,
 * concatenate and project with W_o.
 */

#include "CppNet/layers/attention.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace CppNet
{
    namespace Layers
    {
        MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads,
                                                const std::string& device)
            : embed_dim_(embed_dim), num_heads_(num_heads),
              head_dim_(embed_dim / num_heads), device_(device)
        {
            if (embed_dim_ % num_heads_ != 0)
            {
                throw std::invalid_argument(
                    "embed_dim must be divisible by num_heads");
            }

            float limit = std::sqrt(6.0f / static_cast<float>(embed_dim_ + embed_dim_));

            // Initialize projection weights
            auto init_weight = [&](Eigen::Tensor<float, 2>& w) {
                w.resize(embed_dim_, embed_dim_);
                w.setRandom();
                w = w * w.constant(limit);
            };

            init_weight(W_q_);
            init_weight(W_k_);
            init_weight(W_v_);
            init_weight(W_o_);

            auto init_grad = [&](Eigen::Tensor<float, 2>& g) {
                g.resize(embed_dim_, embed_dim_);
                g.setZero();
            };

            init_grad(grad_W_q_);
            init_grad(grad_W_k_);
            init_grad(grad_W_v_);
            init_grad(grad_W_o_);
        }

        MultiHeadAttention::~MultiHeadAttention() = default;

        const Eigen::Tensor<float, 3> MultiHeadAttention::forward(
            const Eigen::Tensor<float, 3>& query,
            const Eigen::Tensor<float, 3>& key,
            const Eigen::Tensor<float, 3>& value)
        {
            // query/key/value: [batch, seq_len, embed_dim]
            query_cache_ = query;
            key_cache_ = key;
            value_cache_ = value;

            int batch = query.dimension(0);
            int seq_len = query.dimension(1);
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

            Eigen::array<Eigen::IndexPair<int>, 1> contract_last = {Eigen::IndexPair<int>(1, 0)};

            Eigen::Tensor<float, 3> output(batch, seq_len, embed_dim_);
            attention_weights_cache_.resize(batch, seq_len, seq_len);

            for (int n = 0; n < batch; ++n)
            {
                // Extract [seq_len, embed_dim] slices
                Eigen::Tensor<float, 2> Q(seq_len, embed_dim_);
                Eigen::Tensor<float, 2> K(seq_len, embed_dim_);
                Eigen::Tensor<float, 2> V(seq_len, embed_dim_);

                for (int s = 0; s < seq_len; ++s)
                    for (int d = 0; d < embed_dim_; ++d)
                    {
                        Q(s, d) = query(n, s, d);
                        K(s, d) = key(n, s, d);
                        V(s, d) = value(n, s, d);
                    }

                // Project: Q*W_q, K*W_k, V*W_v
                Eigen::Tensor<float, 2> Q_proj = Q.contract(W_q_, contract_last);
                Eigen::Tensor<float, 2> K_proj = K.contract(W_k_, contract_last);
                Eigen::Tensor<float, 2> V_proj = V.contract(W_v_, contract_last);

                // Simplified single-head attention for now:
                // scores = Q_proj * K_proj^T * scale
                Eigen::array<Eigen::IndexPair<int>, 1> contract_inner = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> scores = Q_proj.contract(K_proj, contract_inner);
                scores = scores * scores.constant(scale);

                // Softmax over last dimension
                Eigen::Tensor<float, 2> attn(seq_len, seq_len);
                for (int i = 0; i < seq_len; ++i)
                {
                    float max_val = scores(i, 0);
                    for (int j = 1; j < seq_len; ++j)
                        max_val = std::max(max_val, scores(i, j));

                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; ++j)
                    {
                        attn(i, j) = std::exp(scores(i, j) - max_val);
                        sum += attn(i, j);
                    }
                    for (int j = 0; j < seq_len; ++j)
                        attn(i, j) /= sum;
                }

                // Store attention weights
                for (int i = 0; i < seq_len; ++i)
                    for (int j = 0; j < seq_len; ++j)
                        attention_weights_cache_(n, i, j) = attn(i, j);

                // context = attn * V_proj
                Eigen::Tensor<float, 2> context = attn.contract(V_proj, contract_last);

                // Output projection
                Eigen::Tensor<float, 2> out = context.contract(W_o_, contract_last);

                for (int s = 0; s < seq_len; ++s)
                    for (int d = 0; d < embed_dim_; ++d)
                        output(n, s, d) = out(s, d);
            }

            return output;
        }

        const Eigen::Tensor<float, 3> MultiHeadAttention::backward(
            const Eigen::Tensor<float, 3>& grad_output)
        {
            int batch = grad_output.dimension(0);
            int seq_len = grad_output.dimension(1);
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

            Eigen::Tensor<float, 3> grad_input(batch, seq_len, embed_dim_);
            grad_input.setZero();

            grad_W_q_.setZero();
            grad_W_k_.setZero();
            grad_W_v_.setZero();
            grad_W_o_.setZero();

            // Contraction axes
            Eigen::array<Eigen::IndexPair<int>, 1> contract_last  = {Eigen::IndexPair<int>(1, 0)};
            Eigen::array<Eigen::IndexPair<int>, 1> contract_inner = {Eigen::IndexPair<int>(1, 1)};
            Eigen::array<Eigen::IndexPair<int>, 1> contract_first = {Eigen::IndexPair<int>(0, 0)};

            for (int n = 0; n < batch; ++n)
            {
                Eigen::Tensor<float, 2> grad_slice(seq_len, embed_dim_);
                Eigen::Tensor<float, 2> Q(seq_len, embed_dim_);
                Eigen::Tensor<float, 2> K(seq_len, embed_dim_);
                Eigen::Tensor<float, 2> V(seq_len, embed_dim_);
                Eigen::Tensor<float, 2> attn(seq_len, seq_len);

                for (int s = 0; s < seq_len; ++s)
                    for (int d = 0; d < embed_dim_; ++d)
                    {
                        grad_slice(s, d) = grad_output(n, s, d);
                        Q(s, d) = query_cache_(n, s, d);
                        K(s, d) = key_cache_(n, s, d);
                        V(s, d) = value_cache_(n, s, d);
                    }

                for (int i = 0; i < seq_len; ++i)
                    for (int j = 0; j < seq_len; ++j)
                        attn(i, j) = attention_weights_cache_(n, i, j);

                Eigen::Tensor<float, 2> Q_proj = Q.contract(W_q_, contract_last);
                Eigen::Tensor<float, 2> K_proj = K.contract(W_k_, contract_last);
                Eigen::Tensor<float, 2> V_proj = V.contract(W_v_, contract_last);
                Eigen::Tensor<float, 2> context = attn.contract(V_proj, contract_last);

                // grad_context[s,i] = sum_j grad_slice[s,j] * W_o[i,j]
                Eigen::Tensor<float, 2> grad_context = grad_slice.contract(W_o_, contract_inner);
                // grad_W_o[i,j] += sum_s context[s,i] * grad_slice[s,j]
                grad_W_o_ += context.contract(grad_slice, contract_first);

                // grad_attn[s,t] = sum_d grad_context[s,d] * V_proj[t,d]
                Eigen::Tensor<float, 2> grad_attn = grad_context.contract(V_proj, contract_inner);
                // grad_V_proj[t,d] = sum_s attn[s,t] * grad_context[s,d]
                Eigen::Tensor<float, 2> grad_V_proj = attn.contract(grad_context, contract_first);

                // grad_scores[i,j] = attn[i,j] * (grad_attn[i,j] - dot_i)
                //   where dot_i = sum_k attn[i,k] * grad_attn[i,k]
                Eigen::Tensor<float, 2> grad_scores(seq_len, seq_len);
                for (int i = 0; i < seq_len; ++i)
                {
                    float dot = 0.0f;
                    for (int j = 0; j < seq_len; ++j)
                        dot += grad_attn(i, j) * attn(i, j);
                    for (int j = 0; j < seq_len; ++j)
                        grad_scores(i, j) = attn(i, j) * (grad_attn(i, j) - dot);
                }

                Eigen::Tensor<float, 2> grad_raw = grad_scores * grad_scores.constant(scale);

                // grad_Q_proj[i,k] = sum_j grad_raw[i,j] * K_proj[j,k]
                Eigen::Tensor<float, 2> grad_Q_proj = grad_raw.contract(K_proj, contract_last);
                // grad_K_proj[j,k] = sum_i grad_raw[i,j] * Q_proj[i,k]
                Eigen::Tensor<float, 2> grad_K_proj = grad_raw.contract(Q_proj, contract_first);

                // Q_proj = Q * W_q  →  grad_Q = grad_Q_proj * W_q^T, grad_W_q += Q^T * grad_Q_proj
                Eigen::Tensor<float, 2> grad_Q = grad_Q_proj.contract(W_q_, contract_inner);
                grad_W_q_ += Q.contract(grad_Q_proj, contract_first);

                Eigen::Tensor<float, 2> grad_K = grad_K_proj.contract(W_k_, contract_inner);
                grad_W_k_ += K.contract(grad_K_proj, contract_first);

                Eigen::Tensor<float, 2> grad_V = grad_V_proj.contract(W_v_, contract_inner);
                grad_W_v_ += V.contract(grad_V_proj, contract_first);

                // For self-attention (Q=K=V=input), accumulate all three gradients
                for (int s = 0; s < seq_len; ++s)
                    for (int d = 0; d < embed_dim_; ++d)
                        grad_input(n, s, d) = grad_Q(s, d) + grad_K(s, d) + grad_V(s, d);
            }

            return grad_input;
        }

        void MultiHeadAttention::step(Optimizers::Optimizer& /*optimizer*/, float learning_rate)
        {
            if (!trainable_) return;

            const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

            if (!adam_initialized_) {
                m_q_.resize(W_q_.dimensions()); m_q_.setZero();
                v_q_.resize(W_q_.dimensions()); v_q_.setZero();
                m_k_.resize(W_k_.dimensions()); m_k_.setZero();
                v_k_.resize(W_k_.dimensions()); v_k_.setZero();
                m_v_.resize(W_v_.dimensions()); m_v_.setZero();
                v_v_.resize(W_v_.dimensions()); v_v_.setZero();
                m_o_.resize(W_o_.dimensions()); m_o_.setZero();
                v_o_.resize(W_o_.dimensions()); v_o_.setZero();
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

            int sz = W_q_.size();
            adam_update(W_q_.data(), grad_W_q_.data(), m_q_.data(), v_q_.data(), sz);
            adam_update(W_k_.data(), grad_W_k_.data(), m_k_.data(), v_k_.data(), sz);
            adam_update(W_v_.data(), grad_W_v_.data(), m_v_.data(), v_v_.data(), sz);
            adam_update(W_o_.data(), grad_W_o_.data(), m_o_.data(), v_o_.data(), sz);

            grad_W_q_.setZero();
            grad_W_k_.setZero();
            grad_W_v_.setZero();
            grad_W_o_.setZero();
        }
    }
}