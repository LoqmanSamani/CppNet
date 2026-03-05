/**
 * @file global_pool.hpp
 * @brief Global Average Pooling and Global Max Pooling layers
 *
 * Reduces a 4D tensor [batch, channels, H, W] to [batch, channels]
 * by averaging or taking the max over the spatial dimensions.
 * Commonly used before the final classifier head in CNNs.
 */

#ifndef GLOBAL_POOL_HPP
#define GLOBAL_POOL_HPP

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
         * @class GlobalAvgPool2D
         * @brief Global average pooling over spatial dims
         *
         * input:  [batch, channels, H, W]
         * output: [batch, channels]
         */
        class GlobalAvgPool2D : public Layer
        {
        public:
            explicit GlobalAvgPool2D(const std::string& device = "cpu");
            ~GlobalAvgPool2D() override;

            const Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 4>& input);
            const Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 2>& grad_output);

            bool is_trainable() const override { return false; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

            #ifdef USE_CUDA
                void forward_gpu(const Eigen::Tensor<float, 4>& input,
                                 Eigen::Tensor<float, 2>& output);
                void backward_gpu(const Eigen::Tensor<float, 2>& grad_output,
                                  Eigen::Tensor<float, 4>& grad_input);
                void release_gpu_buffers();
            #endif

        private:
            std::string device_;
            int batch_cache_ = 0;
            int channels_cache_ = 0;
            int height_cache_ = 0;
            int width_cache_ = 0;

            #ifdef USE_CUDA
                float* d_input_ = nullptr;
                float* d_output_ = nullptr;
                float* d_grad_input_ = nullptr;
                float* d_grad_output_ = nullptr;
                std::size_t gpu_buf_in_ = 0;
                std::size_t gpu_buf_out_ = 0;
                bool gpu_initialized_ = false;
            #endif
        };

        /**
         * @class GlobalMaxPool2D
         * @brief Global max pooling over spatial dims
         *
         * input:  [batch, channels, H, W]
         * output: [batch, channels]
         */
        class GlobalMaxPool2D : public Layer
        {
        public:
            explicit GlobalMaxPool2D(const std::string& device = "cpu");
            ~GlobalMaxPool2D() override;

            const Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 4>& input);
            const Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 2>& grad_output);

            bool is_trainable() const override { return false; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

            #ifdef USE_CUDA
                void forward_gpu(const Eigen::Tensor<float, 4>& input,
                                 Eigen::Tensor<float, 2>& output);
                void backward_gpu(const Eigen::Tensor<float, 2>& grad_output,
                                  Eigen::Tensor<float, 4>& grad_input);
                void release_gpu_buffers();
            #endif

        private:
            std::string device_;
            int batch_cache_ = 0;
            int channels_cache_ = 0;
            int height_cache_ = 0;
            int width_cache_ = 0;

            Eigen::Tensor<int, 2> argmax_h_;  // [batch, channels]
            Eigen::Tensor<int, 2> argmax_w_;  // [batch, channels]

            #ifdef USE_CUDA
                float* d_input_ = nullptr;
                float* d_output_ = nullptr;
                int* d_argmax_ = nullptr;
                float* d_grad_input_ = nullptr;
                float* d_grad_output_ = nullptr;
                std::size_t gpu_buf_in_ = 0;
                std::size_t gpu_buf_out_ = 0;
                bool gpu_initialized_ = false;
            #endif
        };
    }
}

#endif // GLOBAL_POOL_HPP
