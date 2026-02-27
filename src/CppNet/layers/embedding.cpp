/**
 * @file embedding.cpp
 * @brief Embedding lookup layer implementation
 *
 * Forward:  weight[input[n][s]] for each (n, s)
 * Backward: scatter-add gradients back to the rows that were looked up
 *
 * Supports three backends:
 *   cpu-eigen : plain C++ loops (baseline)
 *   cpu       : OpenMP-parallelized loops
 *   gpu       : CUDA gather/scatter kernels
 */

#include "CppNet/layers/embedding.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/utils/init.hpp"
#include <cmath>
#include <stdexcept>

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

#ifdef USE_CUDA
            if (device_ == "gpu")
            {
                forward_gpu(input, output);
                return output;
            }
#endif

            if (device_ == "cpu")
            {
                // OpenMP-parallelized forward
#ifdef USE_OPENMP
                #pragma omp parallel for collapse(2) schedule(static)
#endif
                for (int n = 0; n < batch; ++n)
                {
                    for (int s = 0; s < seq_len; ++s)
                    {
                        int idx = input(n, s);
                        for (int d = 0; d < embed_dim_; ++d)
                            output(n, s, d) = weight_(idx, d);
                    }
                }
            }
            else  // cpu-eigen (default)
            {
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
            }

            return output;
        }

        const Eigen::Tensor<float, 3> Embedding::backward(
            const Eigen::Tensor<float, 3>& grad_output)
        {
            // grad_output: [batch, seq_len, embed_dim]
            int batch = grad_output.dimension(0);
            int seq_len = grad_output.dimension(1);

            grad_weight_.setZero();

#ifdef USE_CUDA
            if (device_ == "gpu")
            {
                backward_gpu(grad_output);
                Eigen::Tensor<float, 3> dummy(batch, seq_len, embed_dim_);
                dummy.setZero();
                return dummy;
            }
#endif

            if (device_ == "cpu")
            {
                // OpenMP: parallelize over batch (can't parallel over scatter dim
                // without per-thread accumulators, but batch is safe if no
                // index collisions across different n)
                // Use per-thread local accumulators for safety
#ifdef USE_OPENMP
                int num_threads = omp_get_max_threads();
                std::vector<Eigen::Tensor<float, 2>> thread_grads(num_threads);
                for (auto& g : thread_grads)
                {
                    g.resize(vocab_size_, embed_dim_);
                    g.setZero();
                }

                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    auto& local_grad = thread_grads[tid];
                    #pragma omp for collapse(2) schedule(static)
                    for (int n = 0; n < batch; ++n)
                        for (int s = 0; s < seq_len; ++s)
                        {
                            int idx = input_cache_(n, s);
                            for (int d = 0; d < embed_dim_; ++d)
                                local_grad(idx, d) += grad_output(n, s, d);
                        }
                }

                for (auto& g : thread_grads)
                    for (int i = 0; i < vocab_size_; ++i)
                        for (int d = 0; d < embed_dim_; ++d)
                            grad_weight_(i, d) += g(i, d);
#else
                for (int n = 0; n < batch; ++n)
                    for (int s = 0; s < seq_len; ++s)
                    {
                        int idx = input_cache_(n, s);
                        for (int d = 0; d < embed_dim_; ++d)
                            grad_weight_(idx, d) += grad_output(n, s, d);
                    }
#endif
            }
            else  // cpu-eigen
            {
                for (int n = 0; n < batch; ++n)
                    for (int s = 0; s < seq_len; ++s)
                    {
                        int idx = input_cache_(n, s);
                        for (int d = 0; d < embed_dim_; ++d)
                            grad_weight_(idx, d) += grad_output(n, s, d);
                    }
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

#ifdef USE_CUDA
        void Embedding::forward_gpu(const Eigen::Tensor<int, 2>& input,
                                     Eigen::Tensor<float, 3>& output)
        {
            int batch = input.dimension(0);
            int seq_len = input.dimension(1);
            int batch_seq = batch * seq_len;
            int total = batch_seq * embed_dim_;

            // Allocate device buffers
            int*   d_input  = nullptr;
            float* d_weight = nullptr;
            float* d_output = nullptr;

            cudaMalloc(&d_input,  batch_seq * sizeof(int));
            cudaMalloc(&d_weight, vocab_size_ * embed_dim_ * sizeof(float));
            cudaMalloc(&d_output, total * sizeof(float));

            // Copy input tokens (row-major) and weight table to device
            // Input tensor is [batch, seq_len] in row-major
            cudaMemcpy(d_input, input.data(), batch_seq * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_weight, weight_.data(), vocab_size_ * embed_dim_ * sizeof(float),
                       cudaMemcpyHostToDevice);

            // Launch kernel
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            Kernels::GPU::embedding_forward_kernel<<<blocks, threads>>>(
                d_input, d_weight, d_output, batch_seq, embed_dim_);

            // Copy result back
            cudaMemcpy(output.data(), d_output, total * sizeof(float),
                       cudaMemcpyDeviceToHost);

            cudaFree(d_input);
            cudaFree(d_weight);
            cudaFree(d_output);
        }

        void Embedding::backward_gpu(const Eigen::Tensor<float, 3>& grad_output)
        {
            int batch = grad_output.dimension(0);
            int seq_len = grad_output.dimension(1);
            int batch_seq = batch * seq_len;
            int total = batch_seq * embed_dim_;

            int*   d_input       = nullptr;
            float* d_grad_output = nullptr;
            float* d_grad_weight = nullptr;

            cudaMalloc(&d_input,       batch_seq * sizeof(int));
            cudaMalloc(&d_grad_output, total * sizeof(float));
            cudaMalloc(&d_grad_weight, vocab_size_ * embed_dim_ * sizeof(float));

            cudaMemcpy(d_input, input_cache_.data(), batch_seq * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_output, grad_output.data(), total * sizeof(float),
                       cudaMemcpyHostToDevice);
            // Zero out grad_weight on device
            cudaMemset(d_grad_weight, 0, vocab_size_ * embed_dim_ * sizeof(float));

            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            Kernels::GPU::embedding_backward_kernel<<<blocks, threads>>>(
                d_input, d_grad_output, d_grad_weight, batch_seq, embed_dim_);

            // Copy gradients back to host
            cudaMemcpy(grad_weight_.data(), d_grad_weight,
                       vocab_size_ * embed_dim_ * sizeof(float),
                       cudaMemcpyDeviceToHost);

            cudaFree(d_input);
            cudaFree(d_grad_output);
            cudaFree(d_grad_weight);
        }
#endif
    }
}
