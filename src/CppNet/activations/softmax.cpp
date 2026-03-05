/**
 * @file softmax.cpp
 * @brief Softmax activation implementation
 *
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * Uses the max-subtract trick for numerical stability.
 */

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <omp.h>

#include <Eigen/Dense>
#include "CppNet/activations/softmax.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Activations
    {
        Softmax::Softmax(const std::string& device)
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

        Softmax::~Softmax()
        {
            #ifdef USE_CUDA

                release_gpu_buffers();

            #endif
        }

        void Softmax::set_num_threads(int num_threads)
        {
            if (num_threads > 0)
                omp_set_num_threads(num_threads);
        }

        #ifdef USE_CUDA

            void Softmax::ensure_gpu_buffer_2d(std::size_t num_elements)
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

        void Softmax::ensure_gpu_buffer_4d(std::size_t num_elements)
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

            void Softmax::release_gpu_buffers()
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

        Eigen::Tensor<float, 2> Softmax::forward(const Eigen::Tensor<float, 2>& pre_activation)
        {
            if (pre_activation.size() == 0)
                throw std::runtime_error("Softmax: empty 2D input");

            const int batch = pre_activation.dimension(0);
            const int features = pre_activation.dimension(1);

            output_cache_2d_ = Eigen::Tensor<float, 2>(batch, features);

            if (device_ == "cpu")
            {
                #pragma omp parallel for
                for (int i = 0; i < batch; ++i)
                {
                    float max_val = pre_activation(i, 0);
                    for (int j = 1; j < features; ++j)
                        max_val = std::max(max_val, pre_activation(i, j));

                    float sum = 0.0f;
                    for (int j = 0; j < features; ++j)
                    {
                        output_cache_2d_(i, j) = std::exp(pre_activation(i, j) - max_val);
                        sum += output_cache_2d_(i, j);
                    }

                    for (int j = 0; j < features; ++j)
                        output_cache_2d_(i, j) /= sum;
                }
            }
            else if (device_ == "cpu-eigen")
            {
                for (int i = 0; i < batch; ++i)
                {
                    float max_val = pre_activation(i, 0);
                    for (int j = 1; j < features; ++j)
                        max_val = std::max(max_val, pre_activation(i, j));

                    float sum = 0.0f;
                    for (int j = 0; j < features; ++j)
                    {
                        output_cache_2d_(i, j) = std::exp(pre_activation(i, j) - max_val);
                        sum += output_cache_2d_(i, j);
                    }

                    for (int j = 0; j < features; ++j)
                        output_cache_2d_(i, j) /= sum;
                }
            }
            #ifdef USE_CUDA
                else if (device_ == "gpu")
                {
                    forward_gpu(pre_activation);
                }
            #endif

            return output_cache_2d_;
        }

        Eigen::Tensor<float, 2> Softmax::backward(const Eigen::Tensor<float, 2>& grad_output)
        {
            if (grad_output.size() == 0)
                throw std::runtime_error("Softmax: empty 2D grad");

            const int batch = grad_output.dimension(0);
            const int features = grad_output.dimension(1);

            Eigen::Tensor<float, 2> grad_input = Eigen::Tensor<float, 2>(grad_output.dimensions());

            if (device_ == "cpu")
            {
                #pragma omp parallel for
                for (int i = 0; i < batch; ++i)
                {
                    float dot = 0.0f;
                    for (int j = 0; j < features; ++j)
                        dot += grad_output(i, j) * output_cache_2d_(i, j);

                    for (int j = 0; j < features; ++j)
                        grad_input(i, j) = output_cache_2d_(i, j) * (grad_output(i, j) - dot);
                }
            }
            else if (device_ == "cpu-eigen")
            {
                for (int i = 0; i < batch; ++i)
                {
                    float dot = 0.0f;
                    for (int j = 0; j < features; ++j)
                        dot += grad_output(i, j) * output_cache_2d_(i, j);

                    for (int j = 0; j < features; ++j)
                        grad_input(i, j) = output_cache_2d_(i, j) * (grad_output(i, j) - dot);
                }
            }
            #ifdef USE_CUDA
                else if (device_ == "gpu")
                {
                    backward_gpu(grad_output, grad_input);
                }
            #endif

            return grad_input;
        }

        Eigen::Tensor<float, 4> Softmax::forward(const Eigen::Tensor<float, 4>& pre_activation)
        {
            if (pre_activation.size() == 0)
                throw std::runtime_error("Softmax: empty 4D input");

            const int d0 = pre_activation.dimension(0);
            const int d1 = pre_activation.dimension(1);
            const int d2 = pre_activation.dimension(2);
            const int d3 = pre_activation.dimension(3);

            output_cache_4d_ = Eigen::Tensor<float, 4>(d0, d1, d2, d3);

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(3)
                for (int a = 0; a < d0; ++a)
                    for (int b = 0; b < d1; ++b)
                        for (int c = 0; c < d2; ++c)
                        {
                            float max_val = pre_activation(a, b, c, 0);
                            for (int d = 1; d < d3; ++d)
                                max_val = std::max(max_val, pre_activation(a, b, c, d));

                            float sum = 0.0f;
                            for (int d = 0; d < d3; ++d)
                            {
                                output_cache_4d_(a, b, c, d) = std::exp(pre_activation(a, b, c, d) - max_val);
                                sum += output_cache_4d_(a, b, c, d);
                            }
                            for (int d = 0; d < d3; ++d)
                                output_cache_4d_(a, b, c, d) /= sum;
                        }
            }
            else if (device_ == "cpu-eigen")
            {
                for (int a = 0; a < d0; ++a)
                    for (int b = 0; b < d1; ++b)
                        for (int c = 0; c < d2; ++c)
                        {
                            float max_val = pre_activation(a, b, c, 0);
                            for (int d = 1; d < d3; ++d)
                                max_val = std::max(max_val, pre_activation(a, b, c, d));

                            float sum = 0.0f;
                            for (int d = 0; d < d3; ++d)
                            {
                                output_cache_4d_(a, b, c, d) = std::exp(pre_activation(a, b, c, d) - max_val);
                                sum += output_cache_4d_(a, b, c, d);
                            }
                            for (int d = 0; d < d3; ++d)
                                output_cache_4d_(a, b, c, d) /= sum;
                        }
            }
            #ifdef USE_CUDA
                else if (device_ == "gpu")
                {
                    forward_gpu(pre_activation);
                }
            #endif

            return output_cache_4d_;
        }

        Eigen::Tensor<float, 4> Softmax::backward(const Eigen::Tensor<float, 4>& grad_output)
        {
            if (grad_output.size() == 0)
                throw std::runtime_error("Softmax: empty 4D grad");

            const int d0 = grad_output.dimension(0);
            const int d1 = grad_output.dimension(1);
            const int d2 = grad_output.dimension(2);
            const int d3 = grad_output.dimension(3);

            Eigen::Tensor<float, 4> grad_input = Eigen::Tensor<float, 4>(grad_output.dimensions());

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(3)
                for (int a = 0; a < d0; ++a)
                    for (int b = 0; b < d1; ++b)
                        for (int c = 0; c < d2; ++c)
                        {
                            float dot = 0.0f;
                            for (int d = 0; d < d3; ++d)
                                dot += grad_output(a, b, c, d) * output_cache_4d_(a, b, c, d);

                            for (int d = 0; d < d3; ++d)
                                grad_input(a, b, c, d) = output_cache_4d_(a, b, c, d) * (grad_output(a, b, c, d) - dot);
                        }
            }
            else if (device_ == "cpu-eigen")
            {
                for (int a = 0; a < d0; ++a)
                    for (int b = 0; b < d1; ++b)
                        for (int c = 0; c < d2; ++c)
                        {
                            float dot = 0.0f;
                            for (int d = 0; d < d3; ++d)
                                dot += grad_output(a, b, c, d) * output_cache_4d_(a, b, c, d);

                            for (int d = 0; d < d3; ++d)
                                grad_input(a, b, c, d) = output_cache_4d_(a, b, c, d) * (grad_output(a, b, c, d) - dot);
                        }
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

            void Softmax::forward_gpu(const Eigen::Tensor<float, 2>& pre)
            {
                const int rows = pre.dimension(0);
                const int cols = pre.dimension(1);
                const std::size_t n = pre.size();
                ensure_gpu_buffer_2d(n);

                float* d_in;
                cudaMalloc(&d_in, n * sizeof(float));
                cudaMemcpy(d_in, pre.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                const int block = 256;
                const int smem = block * sizeof(float);

                Kernels::GPU::attention_softmax_forward_kernel<<<rows, block, smem>>>(d_in, d_output_cache_2d_, rows, cols);

                cudaMemcpy(output_cache_2d_.data(), d_output_cache_2d_, n * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_in);
            }

            void Softmax::forward_gpu(const Eigen::Tensor<float, 4>& pre)
            {
                const int d0 = pre.dimension(0);
                const int d1 = pre.dimension(1);
                const int d2 = pre.dimension(2);
                const int d3 = pre.dimension(3);
                const int rows = d0 * d1 * d2;
                const int cols = d3;
                const std::size_t n = pre.size();
                ensure_gpu_buffer_4d(n);

                float* d_in;
                cudaMalloc(&d_in, n * sizeof(float));
                cudaMemcpy(d_in, pre.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                const int block = 256;
                const int smem = block * sizeof(float);

                Kernels::GPU::attention_softmax_forward_kernel<<<rows, block, smem>>>(d_in, d_output_cache_4d_, rows, cols);

                cudaMemcpy(output_cache_4d_.data(), d_output_cache_4d_, n * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_in);
            }

            void Softmax::backward_gpu(const Eigen::Tensor<float, 2>& grad,
                                    Eigen::Tensor<float, 2>& grad_input)
            {
                const int rows = grad.dimension(0);
                const int cols = grad.dimension(1);
                const std::size_t n = grad.size();

                float* d_grad;
                float* d_out;
                cudaMalloc(&d_grad, n * sizeof(float));
                cudaMalloc(&d_out, n * sizeof(float));

                cudaMemcpy(d_grad, grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                const int block = 256;
                const int smem = block * sizeof(float);

                Kernels::GPU::attention_softmax_backward_kernel<<<rows, block, smem>>>(d_grad, d_output_cache_2d_, d_out, rows, cols);

                cudaMemcpy(grad_input.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_grad);
                cudaFree(d_out);
            }

            void Softmax::backward_gpu(const Eigen::Tensor<float, 4>& grad,
                                    Eigen::Tensor<float, 4>& grad_input)
            {
                const int d0 = grad.dimension(0);
                const int d1 = grad.dimension(1);
                const int d2 = grad.dimension(2);
                const int d3 = grad.dimension(3);
                const int rows = d0 * d1 * d2;
                const int cols = d3;
                const std::size_t n = grad.size();

                float* d_grad;
                float* d_out;
                cudaMalloc(&d_grad, n * sizeof(float));
                cudaMalloc(&d_out, n * sizeof(float));

                cudaMemcpy(d_grad, grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                const int block = 256;
                const int smem = block * sizeof(float);

                Kernels::GPU::attention_softmax_backward_kernel<<<rows, block, smem>>>(d_grad, d_output_cache_4d_, d_out, rows, cols);

                cudaMemcpy(grad_input.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_grad);
                cudaFree(d_out);
            }

        #endif

    }
}
