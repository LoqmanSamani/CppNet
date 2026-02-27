/**
 * @file embedding.hpp
 * @brief Embedding lookup layer
 *
 * Maps integer token IDs to dense embedding vectors.
 * Equivalent to a lookup table with learnable weights.
 *
 * Input:  Eigen::Tensor<int, 2>   [batch, seq_len]   (token IDs)
 * Output: Eigen::Tensor<float, 3> [batch, seq_len, embed_dim]
 */

#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class Embedding
         * @brief Token embedding lookup table
         */
        class Embedding : public Layer
        {
        public:
            /**
             * @param vocab_size  Number of unique tokens (rows in embedding table)
             * @param embed_dim   Dimension of each embedding vector
             * @param device      Compute backend
             */
            Embedding(int vocab_size, int embed_dim,
                      const std::string& device = "cpu-eigen");
            ~Embedding();

            /**
             * @brief Forward pass: lookup token IDs in the embedding table
             * @param input [batch, seq_len] integer token IDs (must be in [0, vocab_size))
             * @return [batch, seq_len, embed_dim] dense embeddings
             */
            const Eigen::Tensor<float, 3> forward(const Eigen::Tensor<int, 2>& input);

            /**
             * @brief Backward pass: scatter gradient back to embedding rows
             * @param grad_output [batch, seq_len, embed_dim]
             * @return Dummy zero tensor (no gradient w.r.t. integer indices)
             */
            const Eigen::Tensor<float, 3> backward(const Eigen::Tensor<float, 3>& grad_output);

            bool is_trainable() const override { return trainable_; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

            void freeze() { trainable_ = false; }
            void unfreeze() { trainable_ = true; }

            int get_vocab_size() const { return vocab_size_; }
            int get_embed_dim() const { return embed_dim_; }

            Eigen::Tensor<float, 2>& get_weight() { return weight_; }
            const Eigen::Tensor<float, 2>& get_weight() const { return weight_; }
            const Eigen::Tensor<float, 2>& get_grad_weight() const { return grad_weight_; }

            void set_weight(const Eigen::Tensor<float, 2>& w) { weight_ = w; }

        private:
            int vocab_size_;
            int embed_dim_;
            bool trainable_ = true;
            std::string device_;

            Eigen::Tensor<float, 2> weight_;       // [vocab_size, embed_dim]
            Eigen::Tensor<float, 2> grad_weight_;   // [vocab_size, embed_dim]

            // Cache input indices for backward scatter
            Eigen::Tensor<int, 2> input_cache_;

#ifdef USE_CUDA
            void forward_gpu(const Eigen::Tensor<int, 2>& input,
                             Eigen::Tensor<float, 3>& output);
            void backward_gpu(const Eigen::Tensor<float, 3>& grad_output);
#endif
        };
    }
}

#endif // EMBEDDING_HPP
