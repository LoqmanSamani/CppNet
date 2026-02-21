/**
 * @file embedding.cpp
 * @brief Embedding lookup layer implementation
 *
 * Forward:  weight[input[n][s]] for each (n, s)
 * Backward: scatter-add gradients back to the rows that were looked up
 */

#include "CppNet/layers/embedding.hpp"
#include <cmath>
#include <stdexcept>

namespace CppNet
{
    namespace Layers
    {
        Embedding::Embedding(int vocab_size, int embed_dim,
                             const std::string& device)
            : vocab_size_(vocab_size), embed_dim_(embed_dim), device_(device)
        {
            // Initialize with N(0, 1) scaled by 1/sqrt(embed_dim)
            float scale = 1.0f / std::sqrt(static_cast<float>(embed_dim_));

            weight_.resize(vocab_size_, embed_dim_);
            weight_.setRandom();   // uniform [-1, 1]
            weight_ = weight_ * weight_.constant(scale);

            grad_weight_.resize(vocab_size_, embed_dim_);
            grad_weight_.setZero();
        }

        Embedding::~Embedding() = default;

        const Eigen::Tensor<float, 3> Embedding::forward(
            const Eigen::Tensor<int, 2>& input)
        {
            // input: [batch, seq_len]
            input_cache_ = input;

            int batch = input.dimension(0);
            int seq_len = input.dimension(1);

            Eigen::Tensor<float, 3> output(batch, seq_len, embed_dim_);

            for (int n = 0; n < batch; ++n)
            {
                for (int s = 0; s < seq_len; ++s)
                {
                    int idx = input(n, s);
                    if (idx < 0 || idx >= vocab_size_)
                        throw std::out_of_range(
                            "Embedding: token ID " + std::to_string(idx) +
                            " out of range [0, " + std::to_string(vocab_size_) + ")");

                    for (int d = 0; d < embed_dim_; ++d)
                        output(n, s, d) = weight_(idx, d);
                }
            }

            return output;
        }

        const Eigen::Tensor<float, 3> Embedding::backward(
            const Eigen::Tensor<float, 3>& grad_output)
        {
            // grad_output: [batch, seq_len, embed_dim]
            // Scatter-add to grad_weight_
            int batch = grad_output.dimension(0);
            int seq_len = grad_output.dimension(1);

            grad_weight_.setZero();

            for (int n = 0; n < batch; ++n)
                for (int s = 0; s < seq_len; ++s)
                {
                    int idx = input_cache_(n, s);
                    for (int d = 0; d < embed_dim_; ++d)
                        grad_weight_(idx, d) += grad_output(n, s, d);
                }

            // No meaningful gradient w.r.t. integer inputs
            Eigen::Tensor<float, 3> dummy(batch, seq_len, embed_dim_);
            dummy.setZero();
            return dummy;
        }

        void Embedding::step(Optimizers::Optimizer& /*optimizer*/, float learning_rate)
        {
            // Simple SGD update for the embedding table
            if (!trainable_) return;

            for (int v = 0; v < vocab_size_; ++v)
                for (int d = 0; d < embed_dim_; ++d)
                    weight_(v, d) -= learning_rate * grad_weight_(v, d);

            grad_weight_.setZero();
        }
    }
}
