/**
 * @file leacky_relu.hpp
 * @brief Leaky ReLU activation function
 */

#ifndef LEAKY_RELU_HPP
#define LEAKY_RELU_HPP

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
         * @class LeakyReLU
         * @brief Leaky ReLU: f(x) = x if x > 0, else alpha * x
         *
         * Prevents "dying ReLU" problem by allowing small negative gradients.
         */
        class LeakyReLU final : public Activation
        {
        public:
            /**
             * @param alpha Slope for negative inputs (default 0.01)
             * @param device Compute backend: "cpu-eigen", "cpu", or "gpu"
             */
            explicit LeakyReLU(float alpha = 0.01f, const std::string& device = "cpu-eigen");

            ~LeakyReLU();

            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_activation) override;
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) override;
            Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_activation) override;
            Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output) override;

            static void set_num_threads(int num_threads);
            float get_alpha() const { return alpha_; }

        private:
            float alpha_;
            std::string device_;
            Eigen::Tensor<float, 2> input_cache_2d_;
            Eigen::Tensor<float, 4> input_cache_4d_;
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

#endif // LEAKY_RELU_HPP

