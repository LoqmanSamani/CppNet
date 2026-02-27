/**
 * @file attention.cpp
 * @brief Multi-Head Attention layer implementation
 *
 * Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 * Multi-head: split embed_dim into num_heads, apply attention per head,
 * concatenate and project with W_o.
 *
 * Supports three backends:
 *   cpu-eigen : Eigen tensor contractions
 *   cpu       : OpenMP-parallelized loops
 *   gpu       : CUDA kernels (matmul + custom softmax)
 */

#include "CppNet/layers/attention.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/utils/init.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"
#endif

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

            // Xavier initialization for projection weights
            W_q_ = CppNet::Utils::xavier_uniform(embed_dim_, embed_dim_);
            W_k_ = CppNet::Utils::xavier_uniform(embed_dim_, embed_dim_);
            W_v_ = CppNet::Utils::xavier_uniform(embed_dim_, embed_dim_);
            W_o_ = CppNet::Utils::xavier_uniform(embed_dim_, embed_dim_);

            grad_W_q_ = CppNet::Utils::constant_init(embed_dim_, embed_dim_, 0.0f);
            grad_W_k_ = CppNet::Utils::constant_init(embed_dim_, embed_dim_, 0.0f);
            grad_W_v_ = CppNet::Utils::constant_init(embed_dim_, embed_dim_, 0.0f);
            grad_W_o_ = CppNet::Utils::constant_init(embed_dim_, embed_dim_, 0.0f);
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

            Eigen::Tensor<float, 3> output(batch, seq_len, embed_dim_);
            attention_weights_cache_.resize(batch, seq_len, seq_len);

#ifdef USE_CUDA
            if (device_ == "gpu")
            {
                forward_gpu(query, key, value, output);
                return output;
            }
#endif

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

            if (device_ == "cpu")
            {
                // OpenMP-parallelized forward: parallelize over batch
#ifdef USE_OPENMP
                #pragma omp parallel for schedule(static)
#endif
                for (int n = 0; n < batch; ++n)
                {
                    // Extract [seq_len, embed_dim] slices
                    // Use row-major manual matmul
                    // Q_proj = Q * W_q  (seq_len x embed_dim) * (embed_dim x embed_dim)
                    std::vector<float> Q_proj(seq_len * embed_dim_, 0.0f);
                    std::vector<float> K_proj(seq_len * embed_dim_, 0.0f);
                    std::vector<float> V_proj(seq_len * embed_dim_, 0.0f);

                    for (int s = 0; s < seq_len; ++s)
                        for (int j = 0; j < embed_dim_; ++j)
                        {
                            float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;
                            for (int k = 0; k < embed_dim_; ++k)
                            {
                                sum_q += query(n, s, k) * W_q_(k, j);
                                sum_k += key(n, s, k)   * W_k_(k, j);
                                sum_v += value(n, s, k)  * W_v_(k, j);
                            }
                            Q_proj[s * embed_dim_ + j] = sum_q;
                            K_proj[s * embed_dim_ + j] = sum_k;
                            V_proj[s * embed_dim_ + j] = sum_v;
                        }

                    // scores = Q_proj * K_proj^T * scale  [seq_len x seq_len]
                    std::vector<float> scores(seq_len * seq_len, 0.0f);
                    for (int i = 0; i < seq_len; ++i)
                        for (int j = 0; j < seq_len; ++j)
                        {
                            float sum = 0.0f;
                            for (int k = 0; k < embed_dim_; ++k)
                                sum += Q_proj[i * embed_dim_ + k] * K_proj[j * embed_dim_ + k];
                            scores[i * seq_len + j] = sum * scale;
                        }

                    // Softmax per row
                    std::vector<float> attn(seq_len * seq_len);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        float max_val = scores[i * seq_len];
                        for (int j = 1; j < seq_len; ++j)
                            max_val = std::max(max_val, scores[i * seq_len + j]);
                        float sum = 0.0f;
                        for (int j = 0; j < seq_len; ++j)
                        {
                            attn[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_val);
                            sum += attn[i * seq_len + j];
                        }
                        for (int j = 0; j < seq_len; ++j)
                            attn[i * seq_len + j] /= sum;
                    }

                    // Store attention weights
                    for (int i = 0; i < seq_len; ++i)
                        for (int j = 0; j < seq_len; ++j)
                            attention_weights_cache_(n, i, j) = attn[i * seq_len + j];

                    // context = attn * V_proj  [seq_len x embed_dim]
                    std::vector<float> context(seq_len * embed_dim_, 0.0f);
                    for (int i = 0; i < seq_len; ++i)
                        for (int d = 0; d < embed_dim_; ++d)
                        {
                            float sum = 0.0f;
                            for (int j = 0; j < seq_len; ++j)
                                sum += attn[i * seq_len + j] * V_proj[j * embed_dim_ + d];
                            context[i * embed_dim_ + d] = sum;
                        }

                    // out = context * W_o  [seq_len x embed_dim]
                    for (int s = 0; s < seq_len; ++s)
                        for (int d = 0; d < embed_dim_; ++d)
                        {
                            float sum = 0.0f;
                            for (int k = 0; k < embed_dim_; ++k)
                                sum += context[s * embed_dim_ + k] * W_o_(k, d);
                            output(n, s, d) = sum;
                        }
                }
            }
            else  // cpu-eigen
            {
                Eigen::array<Eigen::IndexPair<int>, 1> contract_last = {Eigen::IndexPair<int>(1, 0)};

                for (int n = 0; n < batch; ++n)
                {
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

                    Eigen::Tensor<float, 2> Q_proj = Q.contract(W_q_, contract_last);
                    Eigen::Tensor<float, 2> K_proj = K.contract(W_k_, contract_last);
                    Eigen::Tensor<float, 2> V_proj = V.contract(W_v_, contract_last);

                    Eigen::array<Eigen::IndexPair<int>, 1> contract_inner = {Eigen::IndexPair<int>(1, 1)};
                    Eigen::Tensor<float, 2> scores = Q_proj.contract(K_proj, contract_inner);
                    scores = scores * scores.constant(scale);

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

                    for (int i = 0; i < seq_len; ++i)
                        for (int j = 0; j < seq_len; ++j)
                            attention_weights_cache_(n, i, j) = attn(i, j);

                    Eigen::Tensor<float, 2> context = attn.contract(V_proj, contract_last);
                    Eigen::Tensor<float, 2> out = context.contract(W_o_, contract_last);

                    for (int s = 0; s < seq_len; ++s)
                        for (int d = 0; d < embed_dim_; ++d)
                            output(n, s, d) = out(s, d);
                }
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

#ifdef USE_CUDA
            if (device_ == "gpu")
            {
                backward_gpu(grad_output, grad_input);
                return grad_input;
            }
#endif

            if (device_ == "cpu")
            {
                // OpenMP: per-thread gradient accumulators to avoid race conditions
#ifdef USE_OPENMP
                int num_threads = omp_get_max_threads();
                std::vector<Eigen::Tensor<float, 2>> t_gWq(num_threads), t_gWk(num_threads),
                                                      t_gWv(num_threads), t_gWo(num_threads);
                for (int t = 0; t < num_threads; ++t)
                {
                    t_gWq[t].resize(embed_dim_, embed_dim_); t_gWq[t].setZero();
                    t_gWk[t].resize(embed_dim_, embed_dim_); t_gWk[t].setZero();
                    t_gWv[t].resize(embed_dim_, embed_dim_); t_gWv[t].setZero();
                    t_gWo[t].resize(embed_dim_, embed_dim_); t_gWo[t].setZero();
                }

                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    #pragma omp for schedule(static)
                    for (int n = 0; n < batch; ++n)
                    {
                        // Manual matmul backward for each sample
                        std::vector<float> Q(seq_len * embed_dim_);
                        std::vector<float> K(seq_len * embed_dim_);
                        std::vector<float> V(seq_len * embed_dim_);
                        std::vector<float> grad_sl(seq_len * embed_dim_);
                        std::vector<float> attn(seq_len * seq_len);

                        for (int s = 0; s < seq_len; ++s)
                            for (int d = 0; d < embed_dim_; ++d)
                            {
                                Q[s * embed_dim_ + d] = query_cache_(n, s, d);
                                K[s * embed_dim_ + d] = key_cache_(n, s, d);
                                V[s * embed_dim_ + d] = value_cache_(n, s, d);
                                grad_sl[s * embed_dim_ + d] = grad_output(n, s, d);
                            }
                        for (int i = 0; i < seq_len; ++i)
                            for (int j = 0; j < seq_len; ++j)
                                attn[i * seq_len + j] = attention_weights_cache_(n, i, j);

                        // Q_proj, K_proj, V_proj
                        std::vector<float> Qp(seq_len * embed_dim_, 0.0f);
                        std::vector<float> Kp(seq_len * embed_dim_, 0.0f);
                        std::vector<float> Vp(seq_len * embed_dim_, 0.0f);
                        for (int s = 0; s < seq_len; ++s)
                            for (int j = 0; j < embed_dim_; ++j)
                            {
                                float sq = 0, sk = 0, sv = 0;
                                for (int k = 0; k < embed_dim_; ++k)
                                {
                                    sq += Q[s*embed_dim_+k] * W_q_(k,j);
                                    sk += K[s*embed_dim_+k] * W_k_(k,j);
                                    sv += V[s*embed_dim_+k] * W_v_(k,j);
                                }
                                Qp[s*embed_dim_+j] = sq;
                                Kp[s*embed_dim_+j] = sk;
                                Vp[s*embed_dim_+j] = sv;
                            }

                        // context = attn * V_proj
                        std::vector<float> ctx(seq_len * embed_dim_, 0.0f);
                        for (int i = 0; i < seq_len; ++i)
                            for (int d = 0; d < embed_dim_; ++d)
                            {
                                float sum = 0;
                                for (int j = 0; j < seq_len; ++j)
                                    sum += attn[i*seq_len+j] * Vp[j*embed_dim_+d];
                                ctx[i*embed_dim_+d] = sum;
                            }

                        // grad_context = grad_slice * W_o^T
                        std::vector<float> g_ctx(seq_len * embed_dim_, 0.0f);
                        for (int s = 0; s < seq_len; ++s)
                            for (int i = 0; i < embed_dim_; ++i)
                            {
                                float sum = 0;
                                for (int j = 0; j < embed_dim_; ++j)
                                    sum += grad_sl[s*embed_dim_+j] * W_o_(i,j);
                                g_ctx[s*embed_dim_+i] = sum;
                            }

                        // grad_W_o += context^T * grad_slice
                        for (int i = 0; i < embed_dim_; ++i)
                            for (int j = 0; j < embed_dim_; ++j)
                            {
                                float sum = 0;
                                for (int s = 0; s < seq_len; ++s)
                                    sum += ctx[s*embed_dim_+i] * grad_sl[s*embed_dim_+j];
                                t_gWo[tid](i,j) += sum;
                            }

                        // grad_attn = g_ctx * Vp^T
                        std::vector<float> g_attn(seq_len * seq_len, 0.0f);
                        for (int i = 0; i < seq_len; ++i)
                            for (int j = 0; j < seq_len; ++j)
                            {
                                float sum = 0;
                                for (int d = 0; d < embed_dim_; ++d)
                                    sum += g_ctx[i*embed_dim_+d] * Vp[j*embed_dim_+d];
                                g_attn[i*seq_len+j] = sum;
                            }

                        // grad_V_proj = attn^T * g_ctx
                        std::vector<float> g_Vp(seq_len * embed_dim_, 0.0f);
                        for (int t = 0; t < seq_len; ++t)
                            for (int d = 0; d < embed_dim_; ++d)
                            {
                                float sum = 0;
                                for (int s = 0; s < seq_len; ++s)
                                    sum += attn[s*seq_len+t] * g_ctx[s*embed_dim_+d];
                                g_Vp[t*embed_dim_+d] = sum;
                            }

                        // softmax backward
                        std::vector<float> g_scores(seq_len * seq_len);
                        for (int i = 0; i < seq_len; ++i)
                        {
                            float dot = 0;
                            for (int j = 0; j < seq_len; ++j)
                                dot += g_attn[i*seq_len+j] * attn[i*seq_len+j];
                            for (int j = 0; j < seq_len; ++j)
                                g_scores[i*seq_len+j] = attn[i*seq_len+j] * (g_attn[i*seq_len+j] - dot);
                        }
                        for (auto& v : g_scores) v *= scale;

                        // grad_Q_proj = g_scores * K_proj
                        std::vector<float> g_Qp(seq_len * embed_dim_, 0.0f);
                        for (int i = 0; i < seq_len; ++i)
                            for (int k = 0; k < embed_dim_; ++k)
                            {
                                float sum = 0;
                                for (int j = 0; j < seq_len; ++j)
                                    sum += g_scores[i*seq_len+j] * Kp[j*embed_dim_+k];
                                g_Qp[i*embed_dim_+k] = sum;
                            }

                        // grad_K_proj = g_scores^T * Q_proj
                        std::vector<float> g_Kp(seq_len * embed_dim_, 0.0f);
                        for (int j = 0; j < seq_len; ++j)
                            for (int k = 0; k < embed_dim_; ++k)
                            {
                                float sum = 0;
                                for (int i = 0; i < seq_len; ++i)
                                    sum += g_scores[i*seq_len+j] * Qp[i*embed_dim_+k];
                                g_Kp[j*embed_dim_+k] = sum;
                            }

                        // grad_Q = g_Qp * W_q^T, grad_W_q += Q^T * g_Qp
                        std::vector<float> g_Q(seq_len * embed_dim_, 0.0f);
                        std::vector<float> g_K(seq_len * embed_dim_, 0.0f);
                        std::vector<float> g_V(seq_len * embed_dim_, 0.0f);

                        for (int s = 0; s < seq_len; ++s)
                            for (int i = 0; i < embed_dim_; ++i)
                            {
                                float sq = 0, sk = 0, sv = 0;
                                for (int j = 0; j < embed_dim_; ++j)
                                {
                                    sq += g_Qp[s*embed_dim_+j] * W_q_(i,j);
                                    sk += g_Kp[s*embed_dim_+j] * W_k_(i,j);
                                    sv += g_Vp[s*embed_dim_+j] * W_v_(i,j);
                                }
                                g_Q[s*embed_dim_+i] = sq;
                                g_K[s*embed_dim_+i] = sk;
                                g_V[s*embed_dim_+i] = sv;
                            }

                        for (int i = 0; i < embed_dim_; ++i)
                            for (int j = 0; j < embed_dim_; ++j)
                            {
                                float sq = 0, sk = 0, sv = 0;
                                for (int s = 0; s < seq_len; ++s)
                                {
                                    sq += Q[s*embed_dim_+i] * g_Qp[s*embed_dim_+j];
                                    sk += K[s*embed_dim_+i] * g_Kp[s*embed_dim_+j];
                                    sv += V[s*embed_dim_+i] * g_Vp[s*embed_dim_+j];
                                }
                                t_gWq[tid](i,j) += sq;
                                t_gWk[tid](i,j) += sk;
                                t_gWv[tid](i,j) += sv;
                            }

                        for (int s = 0; s < seq_len; ++s)
                            for (int d = 0; d < embed_dim_; ++d)
                                grad_input(n, s, d) = g_Q[s*embed_dim_+d] + g_K[s*embed_dim_+d] + g_V[s*embed_dim_+d];
                    }
                }

                // Reduce thread-local gradients
                for (int t = 0; t < num_threads; ++t)
                {
                    grad_W_q_ += t_gWq[t];
                    grad_W_k_ += t_gWk[t];
                    grad_W_v_ += t_gWv[t];
                    grad_W_o_ += t_gWo[t];
                }
#else
                // Plain CPU fallback (same code as cpu-eigen below)
                Eigen::array<Eigen::IndexPair<int>, 1> contract_last  = {Eigen::IndexPair<int>(1, 0)};
                Eigen::array<Eigen::IndexPair<int>, 1> contract_inner = {Eigen::IndexPair<int>(1, 1)};
                Eigen::array<Eigen::IndexPair<int>, 1> contract_first = {Eigen::IndexPair<int>(0, 0)};

                for (int n = 0; n < batch; ++n)
                {
                    Eigen::Tensor<float, 2> grad_slice(seq_len, embed_dim_);
                    Eigen::Tensor<float, 2> Q(seq_len, embed_dim_);
                    Eigen::Tensor<float, 2> K(seq_len, embed_dim_);
                    Eigen::Tensor<float, 2> V(seq_len, embed_dim_);
                    Eigen::Tensor<float, 2> attn_t(seq_len, seq_len);

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
                            attn_t(i, j) = attention_weights_cache_(n, i, j);

                    Eigen::Tensor<float, 2> Q_proj = Q.contract(W_q_, contract_last);
                    Eigen::Tensor<float, 2> K_proj = K.contract(W_k_, contract_last);
                    Eigen::Tensor<float, 2> V_proj = V.contract(W_v_, contract_last);
                    Eigen::Tensor<float, 2> context = attn_t.contract(V_proj, contract_last);

                    Eigen::Tensor<float, 2> grad_context = grad_slice.contract(W_o_, contract_inner);
                    grad_W_o_ += context.contract(grad_slice, contract_first);

                    Eigen::Tensor<float, 2> grad_attn = grad_context.contract(V_proj, contract_inner);
                    Eigen::Tensor<float, 2> grad_V_proj = attn_t.contract(grad_context, contract_first);

                    Eigen::Tensor<float, 2> grad_scores(seq_len, seq_len);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        float dot = 0.0f;
                        for (int j = 0; j < seq_len; ++j)
                            dot += grad_attn(i, j) * attn_t(i, j);
                        for (int j = 0; j < seq_len; ++j)
                            grad_scores(i, j) = attn_t(i, j) * (grad_attn(i, j) - dot);
                    }
                    Eigen::Tensor<float, 2> grad_raw = grad_scores * grad_scores.constant(scale);

                    Eigen::Tensor<float, 2> grad_Q_proj = grad_raw.contract(K_proj, contract_last);
                    Eigen::Tensor<float, 2> grad_K_proj = grad_raw.contract(Q_proj, contract_first);

                    Eigen::Tensor<float, 2> grad_Q = grad_Q_proj.contract(W_q_, contract_inner);
                    grad_W_q_ += Q.contract(grad_Q_proj, contract_first);
                    Eigen::Tensor<float, 2> grad_K = grad_K_proj.contract(W_k_, contract_inner);
                    grad_W_k_ += K.contract(grad_K_proj, contract_first);
                    Eigen::Tensor<float, 2> grad_V = grad_V_proj.contract(W_v_, contract_inner);
                    grad_W_v_ += V.contract(grad_V_proj, contract_first);

                    for (int s = 0; s < seq_len; ++s)
                        for (int d = 0; d < embed_dim_; ++d)
                            grad_input(n, s, d) = grad_Q(s, d) + grad_K(s, d) + grad_V(s, d);
                }
#endif
            }
            else  // cpu-eigen
            {
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

                    Eigen::Tensor<float, 2> grad_context = grad_slice.contract(W_o_, contract_inner);
                    grad_W_o_ += context.contract(grad_slice, contract_first);

                    Eigen::Tensor<float, 2> grad_attn = grad_context.contract(V_proj, contract_inner);
                    Eigen::Tensor<float, 2> grad_V_proj = attn.contract(grad_context, contract_first);

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

                    Eigen::Tensor<float, 2> grad_Q_proj = grad_raw.contract(K_proj, contract_last);
                    Eigen::Tensor<float, 2> grad_K_proj = grad_raw.contract(Q_proj, contract_first);

                    Eigen::Tensor<float, 2> grad_Q = grad_Q_proj.contract(W_q_, contract_inner);
                    grad_W_q_ += Q.contract(grad_Q_proj, contract_first);

                    Eigen::Tensor<float, 2> grad_K = grad_K_proj.contract(W_k_, contract_inner);
                    grad_W_k_ += K.contract(grad_K_proj, contract_first);

                    Eigen::Tensor<float, 2> grad_V = grad_V_proj.contract(W_v_, contract_inner);
                    grad_W_v_ += V.contract(grad_V_proj, contract_first);

                    for (int s = 0; s < seq_len; ++s)
                        for (int d = 0; d < embed_dim_; ++d)
                            grad_input(n, s, d) = grad_Q(s, d) + grad_K(s, d) + grad_V(s, d);
                }
            }

            return grad_input;
        }

        void MultiHeadAttention::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            if (!trainable_) return;

            optimizer.update(W_q_.data(), grad_W_q_.data(), W_q_.size(), learning_rate);
            optimizer.update(W_k_.data(), grad_W_k_.data(), W_k_.size(), learning_rate);
            optimizer.update(W_v_.data(), grad_W_v_.data(), W_v_.size(), learning_rate);
            optimizer.update(W_o_.data(), grad_W_o_.data(), W_o_.size(), learning_rate);

            grad_W_q_.setZero();
            grad_W_k_.setZero();
            grad_W_v_.setZero();
            grad_W_o_.setZero();
        }

#ifdef USE_CUDA
        // ─── GPU Forward ─────────────────────────────────────────────────
        // Processes each batch sample sequentially on the GPU.
        // Uses matmul_kernel for Q/K/V projections and context/output matmuls.
        // Uses custom softmax kernel for attention scores.
        // All matrices stored in ColMajor for matmul_kernel compatibility.
        void MultiHeadAttention::forward_gpu(
            const Eigen::Tensor<float, 3>& query,
            const Eigen::Tensor<float, 3>& key,
            const Eigen::Tensor<float, 3>& value,
            Eigen::Tensor<float, 3>& output)
        {
            int batch = query.dimension(0);
            int S = query.dimension(1);   // seq_len
            int D = embed_dim_;
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

            // Allocate device buffers (per-call, RNN-style)
            float *d_Wq, *d_Wk, *d_Wv, *d_Wo;
            float *d_Q, *d_K, *d_V;
            float *d_Qp, *d_Kp, *d_Vp;
            float *d_scores, *d_attn, *d_ctx, *d_out;

            cudaMalloc(&d_Wq, D * D * sizeof(float));
            cudaMalloc(&d_Wk, D * D * sizeof(float));
            cudaMalloc(&d_Wv, D * D * sizeof(float));
            cudaMalloc(&d_Wo, D * D * sizeof(float));
            cudaMalloc(&d_Q,  S * D * sizeof(float));
            cudaMalloc(&d_K,  S * D * sizeof(float));
            cudaMalloc(&d_V,  S * D * sizeof(float));
            cudaMalloc(&d_Qp, S * D * sizeof(float));
            cudaMalloc(&d_Kp, S * D * sizeof(float));
            cudaMalloc(&d_Vp, S * D * sizeof(float));
            cudaMalloc(&d_scores, S * S * sizeof(float));
            cudaMalloc(&d_attn,   S * S * sizeof(float));
            cudaMalloc(&d_ctx, S * D * sizeof(float));
            cudaMalloc(&d_out, S * D * sizeof(float));

            // Upload weights (ColMajor — Eigen default)
            cudaMemcpy(d_Wq, W_q_.data(), D * D * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Wk, W_k_.data(), D * D * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Wv, W_v_.data(), D * D * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Wo, W_o_.data(), D * D * sizeof(float), cudaMemcpyHostToDevice);

            dim3 block(32, 32);

            for (int n = 0; n < batch; ++n)
            {
                // Extract Q, K, V slices [S, D] and upload (ColMajor)
                Eigen::Tensor<float, 2> Q_h(S, D), K_h(S, D), V_h(S, D);
                for (int s = 0; s < S; ++s)
                    for (int d = 0; d < D; ++d)
                    {
                        Q_h(s, d) = query(n, s, d);
                        K_h(s, d) = key(n, s, d);
                        V_h(s, d) = value(n, s, d);
                    }

                cudaMemcpy(d_Q, Q_h.data(), S * D * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_K, K_h.data(), S * D * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_V, V_h.data(), S * D * sizeof(float), cudaMemcpyHostToDevice);

                // Q_proj = Q * W_q:  [S x D] * [D x D] → [S x D]   (ColMajor)
                dim3 grid_qkv((D + 31) / 32, (S + 31) / 32);
                Kernels::GPU::matmul_kernel<<<grid_qkv, block>>>(d_Q, d_Wq, d_Qp, S, D, D);
                Kernels::GPU::matmul_kernel<<<grid_qkv, block>>>(d_K, d_Wk, d_Kp, S, D, D);
                Kernels::GPU::matmul_kernel<<<grid_qkv, block>>>(d_V, d_Wv, d_Vp, S, D, D);

                // scores = Q_proj * K_proj^T:  [S x D] * [D x S] → [S x S]
                // K_proj^T:  K_proj is [S x D] in ColMajor = column-by-column
                // For A*B^T we can use: matmul_grad_input_kernel reinterpreted,
                // or compute manually. Since matmul_kernel does A*B in ColMajor,
                // and we need [S,D]*[S,D]^T = [S,S], we can use the fact that
                // in ColMajor, K_proj^T is equivalent to [D x S] ColMajor where the
                // underlying data of K_proj [S x D] ColMajor acts as [D x S] RowMajor.
                // Actually, for ColMajor [S,D], the transpose [D,S] shares the same
                // memory layout but with swapped dimensions:
                // A[S,D]*B^T[D,S] = matmul_kernel(A, B, C, S, S, D)
                //   where B is the [S,D] ColMajor data re-interpreted as [D,S] ColMajor
                //   But that doesn't work because B^T[D,S] in ColMajor is different.
                //
                // Use a simple approach: compute scores on CPU from GPU projections
                // and re-upload. Better: use softmax kernel directly.
                //
                // Actually, let's note: for ColMajor layout, A(S,D) is stored as
                // A[row + col*S], and B(S,D) is B[row + col*S].
                // A * B^T: C(i,j) = sum_k A(i,k) * B(j,k)
                //        = sum_k A[i + k*S] * B[j + k*S]
                // This is exactly what matmul_kernel computes when we pass:
                //   matmul_kernel(A_data, B_data, C_data, S, S, D)
                // because matmul_kernel computes C(r,c) = sum_k A(r,k) * B(k,c)
                //   = sum_k A[r + k*M] * B[k + c*K]
                // We need C(i,j) = sum_k A(i,k) * B(j,k) which requires B^T.
                // So we need a temporary transpose. Let me instead download, compute
                // scores with Eigen, and re-upload; this is simpler and works.

                // Download Q_proj, K_proj, V_proj
                Eigen::Tensor<float, 2> Qp_h(S, D), Kp_h(S, D), Vp_h(S, D);
                cudaMemcpy(Qp_h.data(), d_Qp, S * D * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(Kp_h.data(), d_Kp, S * D * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(Vp_h.data(), d_Vp, S * D * sizeof(float), cudaMemcpyDeviceToHost);

                // scores = Q_proj * K_proj^T using Eigen
                Eigen::array<Eigen::IndexPair<int>, 1> contract_inner = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> scores_h = Qp_h.contract(Kp_h, contract_inner);

                // Upload scores and apply scale + softmax on GPU
                cudaMemcpy(d_scores, scores_h.data(), S * S * sizeof(float), cudaMemcpyHostToDevice);

                // Scale
                int total_scores = S * S;
                Kernels::GPU::attention_scale_kernel<<<(total_scores + 255) / 256, 256>>>(
                    d_scores, scale, total_scores);

                // Softmax (each block handles one row)
                int softmax_threads = std::min(256, S);
                // Ensure power of 2 for reductions
                int st = 1;
                while (st < softmax_threads) st <<= 1;
                softmax_threads = st;
                softmax_threads = std::min(softmax_threads, 256);
                Kernels::GPU::attention_softmax_forward_kernel
                    <<<S, softmax_threads, softmax_threads * sizeof(float)>>>(
                        d_scores, d_attn, S, S);

                // Download attention weights to cache
                Eigen::Tensor<float, 2> attn_h(S, S);
                cudaMemcpy(attn_h.data(), d_attn, S * S * sizeof(float), cudaMemcpyDeviceToHost);
                for (int i = 0; i < S; ++i)
                    for (int j = 0; j < S; ++j)
                        attention_weights_cache_(n, i, j) = attn_h(i, j);

                // context = attn * V_proj:  [S x S] * [S x D] → [S x D]
                dim3 grid_ctx((D + 31) / 32, (S + 31) / 32);
                Kernels::GPU::matmul_kernel<<<grid_ctx, block>>>(d_attn, d_Vp, d_ctx, S, D, S);

                // out = context * W_o:  [S x D] * [D x D] → [S x D]
                Kernels::GPU::matmul_kernel<<<grid_qkv, block>>>(d_ctx, d_Wo, d_out, S, D, D);

                // Download output
                Eigen::Tensor<float, 2> out_h(S, D);
                cudaMemcpy(out_h.data(), d_out, S * D * sizeof(float), cudaMemcpyDeviceToHost);
                for (int s = 0; s < S; ++s)
                    for (int d = 0; d < D; ++d)
                        output(n, s, d) = out_h(s, d);
            }

            cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv); cudaFree(d_Wo);
            cudaFree(d_Q);  cudaFree(d_K);  cudaFree(d_V);
            cudaFree(d_Qp); cudaFree(d_Kp); cudaFree(d_Vp);
            cudaFree(d_scores); cudaFree(d_attn);
            cudaFree(d_ctx); cudaFree(d_out);
        }

        // ─── GPU Backward ────────────────────────────────────────────────
        void MultiHeadAttention::backward_gpu(
            const Eigen::Tensor<float, 3>& grad_output,
            Eigen::Tensor<float, 3>& grad_input)
        {
            int batch = grad_output.dimension(0);
            int S = grad_output.dimension(1);
            int D = embed_dim_;
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

            // For GPU backward, we use Eigen on CPU for the matmul backward
            // operations per sample, while leveraging GPU softmax backward.
            // The batch loop is sequential (same as forward_gpu), but the
            // heavy matmuls within each sample benefit from Eigen's SIMD
            // and we use GPU for the softmax backward.

            float *d_grad_attn, *d_attn_cache, *d_grad_scores;
            cudaMalloc(&d_grad_attn,   S * S * sizeof(float));
            cudaMalloc(&d_attn_cache,  S * S * sizeof(float));
            cudaMalloc(&d_grad_scores, S * S * sizeof(float));

            Eigen::array<Eigen::IndexPair<int>, 1> contract_last  = {Eigen::IndexPair<int>(1, 0)};
            Eigen::array<Eigen::IndexPair<int>, 1> contract_inner = {Eigen::IndexPair<int>(1, 1)};
            Eigen::array<Eigen::IndexPair<int>, 1> contract_first = {Eigen::IndexPair<int>(0, 0)};

            for (int n = 0; n < batch; ++n)
            {
                Eigen::Tensor<float, 2> grad_slice(S, D);
                Eigen::Tensor<float, 2> Q(S, D), K(S, D), V(S, D);
                Eigen::Tensor<float, 2> attn(S, S);

                for (int s = 0; s < S; ++s)
                    for (int d = 0; d < D; ++d)
                    {
                        grad_slice(s, d) = grad_output(n, s, d);
                        Q(s, d) = query_cache_(n, s, d);
                        K(s, d) = key_cache_(n, s, d);
                        V(s, d) = value_cache_(n, s, d);
                    }
                for (int i = 0; i < S; ++i)
                    for (int j = 0; j < S; ++j)
                        attn(i, j) = attention_weights_cache_(n, i, j);

                Eigen::Tensor<float, 2> Q_proj = Q.contract(W_q_, contract_last);
                Eigen::Tensor<float, 2> K_proj = K.contract(W_k_, contract_last);
                Eigen::Tensor<float, 2> V_proj = V.contract(W_v_, contract_last);
                Eigen::Tensor<float, 2> context = attn.contract(V_proj, contract_last);

                // grad_context = grad_slice * W_o^T
                Eigen::Tensor<float, 2> grad_context = grad_slice.contract(W_o_, contract_inner);
                grad_W_o_ += context.contract(grad_slice, contract_first);

                // grad_attn = grad_context * V_proj^T
                Eigen::Tensor<float, 2> grad_attn_e = grad_context.contract(V_proj, contract_inner);
                Eigen::Tensor<float, 2> grad_V_proj = attn.contract(grad_context, contract_first);

                // Softmax backward on GPU
                cudaMemcpy(d_grad_attn, grad_attn_e.data(), S * S * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_attn_cache, attn.data(), S * S * sizeof(float), cudaMemcpyHostToDevice);

                int softmax_threads = 1;
                while (softmax_threads < S && softmax_threads < 256) softmax_threads <<= 1;
                Kernels::GPU::attention_softmax_backward_kernel
                    <<<S, softmax_threads, softmax_threads * sizeof(float)>>>(
                        d_grad_attn, d_attn_cache, d_grad_scores, S, S);

                Eigen::Tensor<float, 2> grad_scores(S, S);
                cudaMemcpy(grad_scores.data(), d_grad_scores, S * S * sizeof(float), cudaMemcpyDeviceToHost);

                Eigen::Tensor<float, 2> grad_raw = grad_scores * grad_scores.constant(scale);

                Eigen::Tensor<float, 2> grad_Q_proj = grad_raw.contract(K_proj, contract_last);
                Eigen::Tensor<float, 2> grad_K_proj = grad_raw.contract(Q_proj, contract_first);

                Eigen::Tensor<float, 2> grad_Q = grad_Q_proj.contract(W_q_, contract_inner);
                grad_W_q_ += Q.contract(grad_Q_proj, contract_first);
                Eigen::Tensor<float, 2> grad_K = grad_K_proj.contract(W_k_, contract_inner);
                grad_W_k_ += K.contract(grad_K_proj, contract_first);
                Eigen::Tensor<float, 2> grad_V = grad_V_proj.contract(W_v_, contract_inner);
                grad_W_v_ += V.contract(grad_V_proj, contract_first);

                for (int s = 0; s < S; ++s)
                    for (int d = 0; d < D; ++d)
                        grad_input(n, s, d) = grad_Q(s, d) + grad_K(s, d) + grad_V(s, d);
            }

            cudaFree(d_grad_attn);
            cudaFree(d_attn_cache);
            cudaFree(d_grad_scores);
        }
#endif
    }
}