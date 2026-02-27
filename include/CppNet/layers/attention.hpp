/**
 * @file attention.hpp  
 * @brief Multi-Head Attention layer
 */

#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class MultiHeadAttention
         * @brief Multi-Head Scaled Dot-Product Attention
         *
         * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
         * Multi-head: splits embed_dim into num_heads, applies attention per head,
         * then concatenates and projects.
         *
         * Input:  query, key, value each [batch, seq_len, embed_dim]
         * Output: [batch, seq_len, embed_dim]
         */
        class MultiHeadAttention : public Layer
        {
        public:
            /**
             * @param embed_dim Total embedding dimension
             * @param num_heads Number of attention heads (embed_dim must be divisible by num_heads)
             * @param device Compute backend
             */
            MultiHeadAttention(int embed_dim, int num_heads, const std::string& device = "cpu-eigen");
            ~MultiHeadAttention();

            const Eigen::Tensor<float, 3> forward(
                const Eigen::Tensor<float, 3>& query,
                const Eigen::Tensor<float, 3>& key,
                const Eigen::Tensor<float, 3>& value);

            const Eigen::Tensor<float, 3> backward(const Eigen::Tensor<float, 3>& grad_output);

            bool is_trainable() const override { return trainable_; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

            void freeze() { trainable_ = false; }
            void unfreeze() { trainable_ = true; }

            int get_embed_dim() const { return embed_dim_; }
            int get_num_heads() const { return num_heads_; }

        private:
            int embed_dim_;
            int num_heads_;
            int head_dim_;
            bool trainable_ = true;
            std::string device_;

            Eigen::Tensor<float, 2> W_q_;  // [embed_dim, embed_dim]
            Eigen::Tensor<float, 2> W_k_;  // [embed_dim, embed_dim]
            Eigen::Tensor<float, 2> W_v_;  // [embed_dim, embed_dim]
            Eigen::Tensor<float, 2> W_o_;  // [embed_dim, embed_dim]

            Eigen::Tensor<float, 2> grad_W_q_;
            Eigen::Tensor<float, 2> grad_W_k_;
            Eigen::Tensor<float, 2> grad_W_v_;
            Eigen::Tensor<float, 2> grad_W_o_;

            Eigen::Tensor<float, 3> query_cache_;
            Eigen::Tensor<float, 3> key_cache_;
            Eigen::Tensor<float, 3> value_cache_;
            Eigen::Tensor<float, 3> attention_weights_cache_;

#ifdef USE_CUDA
            void forward_gpu(const Eigen::Tensor<float, 3>& query,
                             const Eigen::Tensor<float, 3>& key,
                             const Eigen::Tensor<float, 3>& value,
                             Eigen::Tensor<float, 3>& output);
            void backward_gpu(const Eigen::Tensor<float, 3>& grad_output,
                              Eigen::Tensor<float, 3>& grad_input);
#endif
        };
    }
}

#endif // ATTENTION_HPP