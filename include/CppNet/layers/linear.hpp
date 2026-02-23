/**
 * @file linear.hpp
 * @brief Linear (fully-connected / dense) layer
 *
 * Computes:
 *   output = input * W + b
 *
 * Parameters:
 *   W (weights): [in_size, out_size]
 *   b (biases):  [out_size]           (optional)
 *
 * Input:  [batch, in_size]
 * Output: [batch, out_size]
 */

#ifndef LINEAR_HPP
#define LINEAR_HPP

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class Linear
         * @brief Fully-connected (dense) layer: y = xW + b
         *
         * Supports three compute backends:
         *   - "cpu-eigen"  Eigen tensor contractions (default)
         *   - "cpu"        OpenMP-parallelised loops
         *   - "gpu"        CUDA kernels (requires USE_CUDA)
         *
         * Weight initialization is delegated to CppNet::Utils::init_weights()
         * and can be selected via the @p weight_init constructor parameter
         * ("xavier", "he", "normal", "zeros", …).
         */
        class Linear : public Layer
        {
        public:
            /**
             * @param in_size     Number of input features
             * @param out_size    Number of output features
             * @param layer_name  Human-readable name (used in error messages)
             * @param trainable   Whether gradients are accumulated
             * @param bias        Whether to include a learnable bias vector
             * @param device      Compute backend: "cpu-eigen", "cpu", or "gpu"
             * @param weight_init Initialization strategy forwarded to init_weights()
             */
            Linear(int in_size, int out_size,
                   std::string layer_name = "Linear",
                   bool trainable = true, bool bias = true,
                   std::string device = "cpu-eigen",
                   std::string weight_init = "xavier");
            ~Linear();

            /// @brief Forward pass: output = input * W + b
            const Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input);

            /// @brief Backward pass: computes grad_input and accumulates param gradients
            const Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output);

            bool is_trainable() const override { return trainable_; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

            void freeze()   { trainable_ = false; }
            void unfreeze() { trainable_ = true; }

            /// @brief Zero-out accumulated parameter gradients
            void reset_grads();

            // ── Accessors ──────────────────────────────────────────────
            int get_input_size()  const { return in_size_; }
            int get_output_size() const { return out_size_; }
            std::string get_layer_name() const { return layer_name_; }
            std::string get_device() const { return device_; }
            bool has_bias() const { return bias_flag_; }

            Eigen::Tensor<float, 2>&       get_weights()       { return weights_; }
            const Eigen::Tensor<float, 2>& get_weights() const { return weights_; }
            Eigen::Tensor<float, 1>&       get_biases()        { return biases_; }
            const Eigen::Tensor<float, 1>& get_biases()  const { return biases_; }

            Eigen::Tensor<float, 2>&       get_grad_weights()       { return grad_weights_; }
            const Eigen::Tensor<float, 2>& get_grad_weights() const { return grad_weights_; }
            Eigen::Tensor<float, 1>&       get_grad_biases()        { return grad_biases_; }
            const Eigen::Tensor<float, 1>& get_grad_biases()  const { return grad_biases_; }

            void set_weights(const Eigen::Tensor<float, 2>& weights) { weights_ = weights; }
            void set_biases(const Eigen::Tensor<float, 1>& biases)   { biases_ = biases; }

            /// @brief Re-initialize weights with a different strategy
            void reinitialize_weights(const std::string& method);

            /// @brief Print a short summary of this layer's configuration
            void print_layer_info() const;

            /// @brief Set the global OpenMP thread count
            static void set_num_threads(int num_threads);

            #ifdef USE_CUDA
            void set_max_batch_size(int max_batch_size);
            void init_gpu_buffers(int max_batch_size);
            void cleanup_gpu_buffers();
            void sync_weights_to_gpu();
            void sync_weights_from_gpu();
            void sync_gradients_from_gpu();

            float* get_d_weights()      { return d_weights_; }
            float* get_d_bias()         { return d_bias_; }
            float* get_d_grad_weights() { return d_grad_weights_; }
            float* get_d_grad_bias()    { return d_grad_bias_; }
            bool is_gpu_initialized() const { return gpu_initialized_; }
            #endif

        private:
            int in_size_;
            int out_size_;
            std::string layer_name_;
            bool trainable_;
            bool bias_flag_;
            std::string device_;
            std::string weight_init_;

            Eigen::Tensor<float, 2> weights_;       // [in_size, out_size]
            Eigen::Tensor<float, 1> biases_;        // [out_size]
            Eigen::Tensor<float, 2> grad_weights_;
            Eigen::Tensor<float, 1> grad_biases_;
            Eigen::Tensor<float, 2> in_cache_;      // cached input for backward

            // GPU output caches (CPU-side storage for return values)
            Eigen::Tensor<float, 2> gpu_output_cache_;
            Eigen::Tensor<float, 2> gpu_grad_input_cache_;

            /// Allocate and zero all parameters and gradient buffers
            void init_params_and_grads();

            // ── Backend-specific helpers ────────────────────────────────
            void forward_cpu(const Eigen::Tensor<float, 2>& input,
                             Eigen::Tensor<float, 2>& output,
                             int batch_size, int input_size, int output_size);

            void forward_eigen(const Eigen::Tensor<float, 2>& input,
                               Eigen::Tensor<float, 2>& output,
                               int batch_size, int input_size, int output_size);

            void backward_cpu(const Eigen::Tensor<float, 2>& grad_output,
                              Eigen::Tensor<float, 2>& grad_input,
                              int batch_size, int output_size, int input_size);

            void backward_eigen(const Eigen::Tensor<float, 2>& grad_output,
                                Eigen::Tensor<float, 2>& grad_input,
                                int batch_size, int output_size, int input_size);

            #ifdef USE_CUDA
            void forward_gpu(const Eigen::Tensor<float, 2>& input,
                             Eigen::Tensor<float, 2>& output,
                             int batch_size, int input_size, int output_size);
            void backward_gpu(int batch_size, int output_size, int input_size);

            float* d_weights_       = nullptr;
            float* d_bias_          = nullptr;
            float* d_input_cache_   = nullptr;
            float* d_output_        = nullptr;
            float* d_grad_weights_  = nullptr;
            float* d_grad_bias_     = nullptr;
            float* d_grad_input_    = nullptr;
            float* d_grad_output_   = nullptr;
            bool   gpu_initialized_ = false;
            int    gpu_max_batch_size_ = 0;
            #endif
        };
    }
}

#endif // LINEAR_HPP