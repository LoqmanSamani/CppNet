/**
 * @file softmax.hpp
 * @brief Softmax activation function
 */

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "CppNet/activations/activation.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace CppNet
{
    namespace Activations
    {
        /**
         * @class Softmax
         * @brief Softmax activation: softmax(x_i) = exp(x_i) / sum(exp(x_j))
         *
         * Applies softmax along the last dimension. Forward normalizes
         * inputs to a probability distribution; backward computes the
         * Jacobian-vector product.
         */
        class Softmax final : public Activation
        {
        public:
            /**
             * @param device Compute backend: "cpu-eigen", "cpu", or "gpu"
             */
            explicit Softmax(const std::string& device = "cpu-eigen");

            ~Softmax();

            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_activation) override;
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) override;
            Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_activation) override;
            Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output) override;

            static void set_num_threads(int num_threads);

        private:
            std::string device_;
            Eigen::Tensor<float, 2> output_cache_2d_;
            Eigen::Tensor<float, 4> output_cache_4d_;

            #ifdef USE_CUDA

                float* d_output_cache_2d_ = nullptr;
                float* d_output_cache_4d_ = nullptr;
                std::size_t gpu_buffer_size_2d_ = 0;
                std::size_t gpu_buffer_size_4d_ = 0;
                bool gpu_initialized_ = false;
                void ensure_gpu_buffer_2d(std::size_t num_elements);
                void ensure_gpu_buffer_4d(std::size_t num_elements);
                void release_gpu_buffers();
                
            #endif

            void forward_gpu(const Eigen::Tensor<float, 2>& pre_activation);
            void backward_gpu(const Eigen::Tensor<float, 2>& grad_output,
                              Eigen::Tensor<float, 2>& grad_input);
            void forward_gpu(const Eigen::Tensor<float, 4>& pre_activation);
            void backward_gpu(const Eigen::Tensor<float, 4>& grad_output,
                              Eigen::Tensor<float, 4>& grad_input);
        };
    }
}

#endif // SOFTMAX_HPP

