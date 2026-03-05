/**
 * @file global_pool.cpp
 * @brief GlobalAvgPool2D and GlobalMaxPool2D implementations
 *
 * Input:  [batch, channels, H, W]
 * Output: [batch, channels]
 */

#include "CppNet/layers/global_pool.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"
#include <limits>
#include <omp.h>
#include <cstring>

namespace CppNet
{
    namespace Layers
    {
        // =====================================================================
        // GlobalAvgPool2D
        // =====================================================================

        GlobalAvgPool2D::GlobalAvgPool2D(const std::string& device)
            : device_(device)
        {
            #ifdef USE_CUDA
                d_input_ = nullptr;
                d_output_ = nullptr;
                d_grad_input_ = nullptr;
                d_grad_output_ = nullptr;
                gpu_buf_in_ = 0;
                gpu_buf_out_ = 0;
                gpu_initialized_ = false;
            #endif
        }

        GlobalAvgPool2D::~GlobalAvgPool2D()
        {
            #ifdef USE_CUDA
                release_gpu_buffers();
            #endif
        }

        #ifdef USE_CUDA

            void GlobalAvgPool2D::release_gpu_buffers()
            {
                if (d_input_) cudaFree(d_input_);
                if (d_output_) cudaFree(d_output_);
                if (d_grad_input_) cudaFree(d_grad_input_);
                if (d_grad_output_) cudaFree(d_grad_output_);
                d_input_ = nullptr;
                d_output_ = nullptr;
                d_grad_input_ = nullptr;
                d_grad_output_ = nullptr;
                gpu_buf_in_ = 0;
                gpu_buf_out_ = 0;
                gpu_initialized_ = false;
            }

            void GlobalAvgPool2D::forward_gpu(const Eigen::Tensor<float, 4>& input,
                                              Eigen::Tensor<float, 2>& output)
            {
                std::size_t n_in = input.size();
                std::size_t n_out = output.size();

                if (!gpu_initialized_ || gpu_buf_in_ < n_in || gpu_buf_out_ < n_out)
                {
                    release_gpu_buffers();
                    cudaMalloc(&d_input_, n_in * sizeof(float));
                    cudaMalloc(&d_output_, n_out * sizeof(float));
                    cudaMalloc(&d_grad_input_, n_in * sizeof(float));
                    cudaMalloc(&d_grad_output_, n_out * sizeof(float));
                    gpu_buf_in_ = n_in;
                    gpu_buf_out_ = n_out;
                    gpu_initialized_ = true;
                }

                cudaMemcpy(d_input_, input.data(), n_in * sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int grid = (static_cast<int>(n_out) + block - 1) / block;

                Kernels::GPU::global_avg_pool2d_forward_kernel<<<grid, block>>>(
                    d_input_, d_output_,
                    batch_cache_, channels_cache_, height_cache_, width_cache_);

                cudaMemcpy(output.data(), d_output_, n_out * sizeof(float), cudaMemcpyDeviceToHost);
            }

            void GlobalAvgPool2D::backward_gpu(const Eigen::Tensor<float, 2>& grad_output,
                                               Eigen::Tensor<float, 4>& grad_input)
            {
                std::size_t n_in = grad_input.size();
                std::size_t n_out = grad_output.size();

                cudaMemcpy(d_grad_output_, grad_output.data(), n_out * sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int grid = (static_cast<int>(n_in) + block - 1) / block;

                Kernels::GPU::global_avg_pool2d_backward_kernel<<<grid, block>>>(
                    d_grad_output_, d_grad_input_,
                    batch_cache_, channels_cache_, height_cache_, width_cache_);

                cudaMemcpy(grad_input.data(), d_grad_input_, n_in * sizeof(float), cudaMemcpyDeviceToHost);
            }

        #endif

        const Eigen::Tensor<float, 2> GlobalAvgPool2D::forward(
            const Eigen::Tensor<float, 4>& input)
        {
            batch_cache_    = input.dimension(0);
            channels_cache_ = input.dimension(1);
            height_cache_   = input.dimension(2);
            width_cache_    = input.dimension(3);

            int spatial = height_cache_ * width_cache_;
            float inv_spatial = 1.0f / static_cast<float>(spatial);

            Eigen::Tensor<float, 2> output(batch_cache_, channels_cache_);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    forward_gpu(input, output);
                    return output;
                }
            #endif

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(2)
                for (int n = 0; n < batch_cache_; ++n)
                    for (int c = 0; c < channels_cache_; ++c)
                    {
                        float sum = 0.0f;
                        for (int h = 0; h < height_cache_; ++h)
                            for (int w = 0; w < width_cache_; ++w)
                                sum += input(n, c, h, w);
                        output(n, c) = sum * inv_spatial;
                    }
            }
            else
            {
                for (int n = 0; n < batch_cache_; ++n)
                    for (int c = 0; c < channels_cache_; ++c)
                    {
                        float sum = 0.0f;
                        for (int h = 0; h < height_cache_; ++h)
                            for (int w = 0; w < width_cache_; ++w)
                                sum += input(n, c, h, w);
                        output(n, c) = sum * inv_spatial;
                    }
            }

            return output;
        }

        const Eigen::Tensor<float, 4> GlobalAvgPool2D::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            int spatial = height_cache_ * width_cache_;
            float inv_spatial = 1.0f / static_cast<float>(spatial);

            Eigen::Tensor<float, 4> grad_input(batch_cache_, channels_cache_,
                                                 height_cache_, width_cache_);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    backward_gpu(grad_output, grad_input);
                    return grad_input;
                }
            #endif

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(2)
                for (int n = 0; n < batch_cache_; ++n)
                    for (int c = 0; c < channels_cache_; ++c)
                    {
                        float g = grad_output(n, c) * inv_spatial;
                        for (int h = 0; h < height_cache_; ++h)
                            for (int w = 0; w < width_cache_; ++w)
                                grad_input(n, c, h, w) = g;
                    }
            }
            else
            {
                for (int n = 0; n < batch_cache_; ++n)
                    for (int c = 0; c < channels_cache_; ++c)
                    {
                        float g = grad_output(n, c) * inv_spatial;
                        for (int h = 0; h < height_cache_; ++h)
                            for (int w = 0; w < width_cache_; ++w)
                                grad_input(n, c, h, w) = g;
                    }
            }

            return grad_input;
        }

        void GlobalAvgPool2D::step(Optimizers::Optimizer&, float) {}

        // =====================================================================
        // GlobalMaxPool2D
        // =====================================================================

        GlobalMaxPool2D::GlobalMaxPool2D(const std::string& device)
            : device_(device)
        {
            #ifdef USE_CUDA
                d_input_ = nullptr;
                d_output_ = nullptr;
                d_argmax_ = nullptr;
                d_grad_input_ = nullptr;
                d_grad_output_ = nullptr;
                gpu_buf_in_ = 0;
                gpu_buf_out_ = 0;
                gpu_initialized_ = false;
            #endif
        }

        GlobalMaxPool2D::~GlobalMaxPool2D()
        {
            #ifdef USE_CUDA
                release_gpu_buffers();
            #endif
        }

        #ifdef USE_CUDA

            void GlobalMaxPool2D::release_gpu_buffers()
            {
                if (d_input_) cudaFree(d_input_);
                if (d_output_) cudaFree(d_output_);
                if (d_argmax_) cudaFree(d_argmax_);
                if (d_grad_input_) cudaFree(d_grad_input_);
                if (d_grad_output_) cudaFree(d_grad_output_);
                d_input_ = nullptr;
                d_output_ = nullptr;
                d_argmax_ = nullptr;
                d_grad_input_ = nullptr;
                d_grad_output_ = nullptr;
                gpu_buf_in_ = 0;
                gpu_buf_out_ = 0;
                gpu_initialized_ = false;
            }

            void GlobalMaxPool2D::forward_gpu(const Eigen::Tensor<float, 4>& input,
                                              Eigen::Tensor<float, 2>& output)
            {
                std::size_t n_in = input.size();
                std::size_t n_out = output.size();

                if (!gpu_initialized_ || gpu_buf_in_ < n_in || gpu_buf_out_ < n_out)
                {
                    release_gpu_buffers();
                    cudaMalloc(&d_input_, n_in * sizeof(float));
                    cudaMalloc(&d_output_, n_out * sizeof(float));
                    cudaMalloc(&d_argmax_, n_out * sizeof(int));
                    cudaMalloc(&d_grad_input_, n_in * sizeof(float));
                    cudaMalloc(&d_grad_output_, n_out * sizeof(float));
                    gpu_buf_in_ = n_in;
                    gpu_buf_out_ = n_out;
                    gpu_initialized_ = true;
                }

                cudaMemcpy(d_input_, input.data(), n_in * sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int grid = (static_cast<int>(n_out) + block - 1) / block;

                Kernels::GPU::global_max_pool2d_forward_kernel<<<grid, block>>>(
                    d_input_, d_output_, d_argmax_,
                    batch_cache_, channels_cache_, height_cache_, width_cache_);

                cudaMemcpy(output.data(), d_output_, n_out * sizeof(float), cudaMemcpyDeviceToHost);
            }

            void GlobalMaxPool2D::backward_gpu(const Eigen::Tensor<float, 2>& grad_output,
                                               Eigen::Tensor<float, 4>& grad_input)
            {
                std::size_t n_in = grad_input.size();
                std::size_t n_out = grad_output.size();

                cudaMemset(d_grad_input_, 0, n_in * sizeof(float));
                cudaMemcpy(d_grad_output_, grad_output.data(), n_out * sizeof(float), cudaMemcpyHostToDevice);

                int spatial = height_cache_ * width_cache_;
                int block = 256;
                int grid = (static_cast<int>(n_out) + block - 1) / block;

                Kernels::GPU::global_max_pool2d_backward_kernel<<<grid, block>>>(
                    d_grad_output_, d_argmax_, d_grad_input_,
                    batch_cache_, channels_cache_, spatial);

                cudaMemcpy(grad_input.data(), d_grad_input_, n_in * sizeof(float), cudaMemcpyDeviceToHost);
            }

        #endif

        const Eigen::Tensor<float, 2> GlobalMaxPool2D::forward(
            const Eigen::Tensor<float, 4>& input)
        {
            batch_cache_    = input.dimension(0);
            channels_cache_ = input.dimension(1);
            height_cache_   = input.dimension(2);
            width_cache_    = input.dimension(3);

            argmax_h_.resize(batch_cache_, channels_cache_);
            argmax_w_.resize(batch_cache_, channels_cache_);

            Eigen::Tensor<float, 2> output(batch_cache_, channels_cache_);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    forward_gpu(input, output);
                    return output;
                }
            #endif

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(2)
                for (int n = 0; n < batch_cache_; ++n)
                    for (int c = 0; c < channels_cache_; ++c)
                    {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int best_h = 0, best_w = 0;

                        for (int h = 0; h < height_cache_; ++h)
                            for (int w = 0; w < width_cache_; ++w)
                                if (input(n, c, h, w) > max_val)
                                {
                                    max_val = input(n, c, h, w);
                                    best_h = h;
                                    best_w = w;
                                }

                        output(n, c) = max_val;
                        argmax_h_(n, c) = best_h;
                        argmax_w_(n, c) = best_w;
                    }
            }
            else
            {
                for (int n = 0; n < batch_cache_; ++n)
                    for (int c = 0; c < channels_cache_; ++c)
                    {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int best_h = 0, best_w = 0;

                        for (int h = 0; h < height_cache_; ++h)
                            for (int w = 0; w < width_cache_; ++w)
                                if (input(n, c, h, w) > max_val)
                                {
                                    max_val = input(n, c, h, w);
                                    best_h = h;
                                    best_w = w;
                                }

                        output(n, c) = max_val;
                        argmax_h_(n, c) = best_h;
                        argmax_w_(n, c) = best_w;
                    }
            }

            return output;
        }

        const Eigen::Tensor<float, 4> GlobalMaxPool2D::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            Eigen::Tensor<float, 4> grad_input(batch_cache_, channels_cache_,
                                                 height_cache_, width_cache_);
            grad_input.setZero();

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    backward_gpu(grad_output, grad_input);
                    return grad_input;
                }
            #endif

            for (int n = 0; n < batch_cache_; ++n)
                for (int c = 0; c < channels_cache_; ++c)
                    grad_input(n, c, argmax_h_(n, c), argmax_w_(n, c)) =
                        grad_output(n, c);

            return grad_input;
        }

        void GlobalMaxPool2D::step(Optimizers::Optimizer&, float) {}
    }
}
