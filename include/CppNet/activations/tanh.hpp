/**
 * @file tanh.hpp
 * @brief Tanh activation function
 */

#ifndef TANH_ACTIVATION_HPP
#define TANH_ACTIVATION_HPP

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
         * @class Tanh
         * @brief Tanh activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
         *
         * Output range is (-1, 1). Backward: grad * (1 - tanh(x)^2).
         */
        class Tanh final : public Activation
        {
        public:
            explicit Tanh(const std::string& device = "cpu-eigen");

            ~Tanh();

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

#endif // TANH_ACTIVATION_HPP
