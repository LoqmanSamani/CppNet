/**
 * @file tanh.cpp
 * @brief Tanh activation implementation
 *
 * tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * Backward: dL/dx = dL/dy * (1 - tanh(x)^2)
 */

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <omp.h>

#include <Eigen/Dense>
#include "CppNet/activations/tanh.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Activations
    {
        Tanh::Tanh(const std::string& device)
            : device_(device)
        {
            #ifdef USE_CUDA

                d_output_cache_2d_ = nullptr;
                d_output_cache_4d_ = nullptr;
                gpu_buffer_size_2d_ = 0;
                gpu_buffer_size_4d_ = 0;
                gpu_initialized_ = false;

            #endif
        }

        Tanh::~Tanh()
        {
            #ifdef USE_CUDA

                release_gpu_buffers();

            #endif
        }

        void Tanh::set_num_threads(int num_threads)
        {
            if (num_threads > 0)
                omp_set_num_threads(num_threads);
        }

        #ifdef USE_CUDA

            void Tanh::ensure_gpu_buffer_2d(std::size_t num_elements)
            {
                if (gpu_buffer_size_2d_ >= num_elements)
                    return;

                if (d_output_cache_2d_)
                    cudaFree(d_output_cache_2d_);

                cudaError_t err = cudaMalloc(&d_output_cache_2d_, num_elements * sizeof(float));
                
                if (err != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorString(err));

                gpu_buffer_size_2d_ = num_elements;
                gpu_initialized_ = true;
            }

        void Tanh::ensure_gpu_buffer_4d(std::size_t num_elements)
        {
            if (gpu_buffer_size_4d_ >= num_elements)
                return;

            if (d_output_cache_4d_)
                cudaFree(d_output_cache_4d_);

            cudaError_t err = cudaMalloc(&d_output_cache_4d_, num_elements * sizeof(float));
            
            if (err != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(err));

            gpu_buffer_size_4d_ = num_elements;
            gpu_initialized_ = true;
        }

            void Tanh::release_gpu_buffers()
            {
                if (d_output_cache_2d_)
                    cudaFree(d_output_cache_2d_);
                    
                if (d_output_cache_4d_)
                    cudaFree(d_output_cache_4d_);

                d_output_cache_2d_ = nullptr;
                d_output_cache_4d_ = nullptr;
                gpu_buffer_size_2d_ = 0;
                gpu_buffer_size_4d_ = 0;
                gpu_initialized_ = false;
            }

        #endif

        Eigen::Tensor<float, 2> Tanh::forward(const Eigen::Tensor<float, 2>& pre_activation)
        {
            if (pre_activation.size() == 0)
                throw std::runtime_error("Tanh: empty 2D input");

            const int rows = pre_activation.dimension(0);
            const int cols = pre_activation.dimension(1);

            output_cache_2d_ = Eigen::Tensor<float, 2>(rows, cols);

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        output_cache_2d_(i, j) = std::tanh(pre_activation(i, j));
            }
            else if (device_ == "cpu-eigen")
            {
                output_cache_2d_ = pre_activation.unaryExpr([](float x) { return std::tanh(x); });
            }
            #ifdef USE_CUDA
                else if (device_ == "gpu")
                {
                    forward_gpu(pre_activation);
                }
            #endif

            return output_cache_2d_;
        }

        Eigen::Tensor<float, 2> Tanh::backward(const Eigen::Tensor<float, 2>& grad_output)
        {
            if (grad_output.size() == 0)
                throw std::runtime_error("Tanh: empty 2D grad");

            Eigen::Tensor<float, 2> grad_input = Eigen::Tensor<float, 2>(grad_output.dimensions());

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(2)
                    for (int i = 0; i < grad_output.dimension(0); ++i)
                        for (int j = 0; j < grad_output.dimension(1); ++j)
                        {
                            float t = output_cache_2d_(i, j);
                            grad_input(i, j) = grad_output(i, j) * (1.0f - t * t);
                        }
            }
            else if (device_ == "cpu-eigen")
            {
                auto one_minus_sq = output_cache_2d_.constant(1.0f) - output_cache_2d_ * output_cache_2d_;
                grad_input = grad_output * one_minus_sq;
            }
            #ifdef USE_CUDA
                else if (device_ == "gpu")
                {
                    backward_gpu(grad_output, grad_input);
                }
            #endif

            return grad_input;
        }

        Eigen::Tensor<float, 4> Tanh::forward(const Eigen::Tensor<float, 4>& pre_activation)
        {
            if (pre_activation.size() == 0)
                throw std::runtime_error("Tanh: empty 4D input");

            const int b = pre_activation.dimension(0);
            const int c = pre_activation.dimension(1);
            const int h = pre_activation.dimension(2);
            const int w = pre_activation.dimension(3);

            output_cache_4d_ = Eigen::Tensor<float, 4>(b, c, h, w);

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(4)
                    for (int n = 0; n < b; ++n)
                        for (int ch = 0; ch < c; ++ch)
                            for (int i = 0; i < h; ++i)
                                for (int j = 0; j < w; ++j)
                                    output_cache_4d_(n, ch, i, j) = std::tanh(pre_activation(n, ch, i, j));
            }
            else if (device_ == "cpu-eigen")
            {
                output_cache_4d_ = pre_activation.unaryExpr([](float x) { return std::tanh(x); });
            }
            #ifdef USE_CUDA
                else if (device_ == "gpu")
                {
                    forward_gpu(pre_activation);
                }
            #endif

            return output_cache_4d_;
        }

        Eigen::Tensor<float, 4> Tanh::backward(const Eigen::Tensor<float, 4>& grad_output)
        {
            if (grad_output.size() == 0)
                throw std::runtime_error("Tanh: empty 4D grad");

            Eigen::Tensor<float, 4> grad_input = Eigen::Tensor<float, 4>(grad_output.dimensions());

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(4)
                    for (int n = 0; n < grad_output.dimension(0); ++n)
                        for (int c = 0; c < grad_output.dimension(1); ++c)
                            for (int i = 0; i < grad_output.dimension(2); ++i)
                                for (int j = 0; j < grad_output.dimension(3); ++j)
                                {
                                    float t = output_cache_4d_(n, c, i, j);
                                    grad_input(n, c, i, j) = grad_output(n, c, i, j) * (1.0f - t * t);
                                }
            }
            else if (device_ == "cpu-eigen")
            {
                auto one_minus_sq = output_cache_4d_.constant(1.0f) - output_cache_4d_ * output_cache_4d_;
                grad_input = grad_output * one_minus_sq;
            }
            #ifdef USE_CUDA
                else if (device_ == "gpu")
                {
                    backward_gpu(grad_output, grad_input);
                }
            #endif

            return grad_input;
        }

        #ifdef USE_CUDA
    
            void Tanh::forward_gpu(const Eigen::Tensor<float, 2>& pre)
            {
                const std::size_t n = pre.size();
                ensure_gpu_buffer_2d(n);

                float* d_in;
                cudaMalloc(&d_in, n * sizeof(float));
                cudaMemcpy(d_in, pre.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                const int block = 256;
                const int grid = (n + block - 1) / block;

                Kernels::GPU::tanh_kernel<<<grid, block>>>(d_in, d_output_cache_2d_, n);

                cudaMemcpy(output_cache_2d_.data(), d_output_cache_2d_, n * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_in);
            }

            void Tanh::forward_gpu(const Eigen::Tensor<float, 4>& pre)
            {
                const std::size_t n = pre.size();
                ensure_gpu_buffer_4d(n);

                float* d_in;
                cudaMalloc(&d_in, n * sizeof(float));
                cudaMemcpy(d_in, pre.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                const int block = 256;
                const int grid = (n + block - 1) / block;

                Kernels::GPU::tanh_kernel<<<grid, block>>>(d_in, d_output_cache_4d_, n);

                cudaMemcpy(output_cache_4d_.data(), d_output_cache_4d_, n * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_in);
            }

            void Tanh::backward_gpu(const Eigen::Tensor<float, 2>& grad,
                                    Eigen::Tensor<float, 2>& grad_input)
            {
                const std::size_t n = grad.size();

                float* d_grad;
                float* d_out;
                cudaMalloc(&d_grad, n * sizeof(float));
                cudaMalloc(&d_out, n * sizeof(float));

                cudaMemcpy(d_grad, grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                const int block = 256;
                const int grid = (n + block - 1) / block;

                Kernels::GPU::tanh_grad_kernel<<<grid, block>>>(d_grad, d_output_cache_2d_, d_out, n);

                cudaMemcpy(grad_input.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_grad);
                cudaFree(d_out);
            }

            void Tanh::backward_gpu(const Eigen::Tensor<float, 4>& grad,
                                    Eigen::Tensor<float, 4>& grad_input)
            {
                const std::size_t n = grad.size();

                float* d_grad;
                float* d_out;
                cudaMalloc(&d_grad, n * sizeof(float));
                cudaMalloc(&d_out, n * sizeof(float));

                cudaMemcpy(d_grad, grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                const int block = 256;
                const int grid = (n + block - 1) / block;

                Kernels::GPU::tanh_grad_kernel<<<grid, block>>>(d_grad, d_output_cache_4d_, d_out, n);

                cudaMemcpy(grad_input.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_grad);
                cudaFree(d_out);
            }

        #endif
    }
}
