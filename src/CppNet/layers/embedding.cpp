/**
 * @file embedding.cpp
 * @brief Embedding lookup layer implementation
 *
 * Forward:  weight[input[n][s]] for each (n, s)
 * Backward: scatter-add gradients back to the rows that were looked up
 */

#include "CppNet/layers/embedding.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/utils/init.hpp"
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
            // Uniform[-1/sqrt(d), 1/sqrt(d)] initialization
            float scale = 1.0f / std::sqrt(static_cast<float>(embed_dim_));
            weight_ = CppNet::Utils::uniform_init(vocab_size_, embed_dim_, -scale, scale);

            grad_weight_ = CppNet::Utils::constant_init(vocab_size_, embed_dim_, 0.0f);
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

        void Embedding::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            if (!trainable_) return;

            optimizer.update(weight_.data(), grad_weight_.data(),
                             weight_.size(), learning_rate);

            grad_weight_.setZero();
        }
    }
}
