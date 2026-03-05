/**
 * @file batch_norm.cpp
 * @brief Batch Normalization layer implementation
 *
 * Training:   x_hat = (x - mean) / sqrt(var + eps)
 *             y = gamma * x_hat + beta
 *             running_mean = (1-m)*running_mean + m*batch_mean
 *             running_var  = (1-m)*running_var  + m*batch_var
 *
 * Inference:  x_hat = (x - running_mean) / sqrt(running_var + eps)
 *             y = gamma * x_hat + beta
 */

#include "CppNet/layers/batch_norm.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"
#include <cmath>
#include <stdexcept>
#include <omp.h>

namespace CppNet
{
    namespace Layers
    {
        BatchNorm::BatchNorm(int num_features, float momentum, float eps,
                             const std::string& device)
            : num_features_(num_features), momentum_(momentum), eps_(eps),
              device_(device)
        {
            gamma_.resize(num_features_);
            beta_.resize(num_features_);
            gamma_.setConstant(1.0f);
            beta_.setZero();

            grad_gamma_.resize(num_features_);
            grad_beta_.resize(num_features_);
            grad_gamma_.setZero();
            grad_beta_.setZero();

            running_mean_.resize(num_features_);
            running_var_.resize(num_features_);
            running_mean_.setZero();
            running_var_.setConstant(1.0f);

            batch_mean_.resize(num_features_);
            batch_var_.resize(num_features_);

            #ifdef USE_CUDA
                d_input_ = nullptr;
                d_output_ = nullptr;
                d_x_hat_ = nullptr;
                d_gamma_ = nullptr;
                d_beta_ = nullptr;
                d_batch_mean_ = nullptr;
                d_batch_var_ = nullptr;
                d_running_mean_ = nullptr;
                d_running_var_ = nullptr;
                d_grad_input_ = nullptr;
                d_grad_output_ = nullptr;
                d_grad_gamma_ = nullptr;
                d_grad_beta_ = nullptr;
                gpu_buffer_size_ = 0;
                gpu_initialized_ = false;
            #endif
        }

        BatchNorm::~BatchNorm()
        {
            #ifdef USE_CUDA
                release_gpu_buffers();
            #endif
        }

        #ifdef USE_CUDA

            void BatchNorm::ensure_gpu_buffers(int batch, int features)
            {
                std::size_t n2d = static_cast<std::size_t>(batch) * features;
                std::size_t n1d = static_cast<std::size_t>(features);

                if (gpu_initialized_ && gpu_buffer_size_ >= n2d)
                    return;

                release_gpu_buffers();

                cudaMalloc(&d_input_, n2d * sizeof(float));
                cudaMalloc(&d_output_, n2d * sizeof(float));
                cudaMalloc(&d_x_hat_, n2d * sizeof(float));
                cudaMalloc(&d_grad_input_, n2d * sizeof(float));
                cudaMalloc(&d_grad_output_, n2d * sizeof(float));

                cudaMalloc(&d_gamma_, n1d * sizeof(float));
                cudaMalloc(&d_beta_, n1d * sizeof(float));
                cudaMalloc(&d_batch_mean_, n1d * sizeof(float));
                cudaMalloc(&d_batch_var_, n1d * sizeof(float));
                cudaMalloc(&d_running_mean_, n1d * sizeof(float));
                cudaMalloc(&d_running_var_, n1d * sizeof(float));
                cudaMalloc(&d_grad_gamma_, n1d * sizeof(float));
                cudaMalloc(&d_grad_beta_, n1d * sizeof(float));

                gpu_buffer_size_ = n2d;
                gpu_initialized_ = true;
            }

            void BatchNorm::release_gpu_buffers()
            {
                if (d_input_) cudaFree(d_input_);
                if (d_output_) cudaFree(d_output_);
                if (d_x_hat_) cudaFree(d_x_hat_);
                if (d_gamma_) cudaFree(d_gamma_);
                if (d_beta_) cudaFree(d_beta_);
                if (d_batch_mean_) cudaFree(d_batch_mean_);
                if (d_batch_var_) cudaFree(d_batch_var_);
                if (d_running_mean_) cudaFree(d_running_mean_);
                if (d_running_var_) cudaFree(d_running_var_);
                if (d_grad_input_) cudaFree(d_grad_input_);
                if (d_grad_output_) cudaFree(d_grad_output_);
                if (d_grad_gamma_) cudaFree(d_grad_gamma_);
                if (d_grad_beta_) cudaFree(d_grad_beta_);

                d_input_ = nullptr;
                d_output_ = nullptr;
                d_x_hat_ = nullptr;
                d_gamma_ = nullptr;
                d_beta_ = nullptr;
                d_batch_mean_ = nullptr;
                d_batch_var_ = nullptr;
                d_running_mean_ = nullptr;
                d_running_var_ = nullptr;
                d_grad_input_ = nullptr;
                d_grad_output_ = nullptr;
                d_grad_gamma_ = nullptr;
                d_grad_beta_ = nullptr;
                gpu_buffer_size_ = 0;
                gpu_initialized_ = false;
            }

            void BatchNorm::forward_gpu(const Eigen::Tensor<float, 2>& input,
                                        Eigen::Tensor<float, 2>& output)
            {
                int batch = input.dimension(0);
                int features = input.dimension(1);
                std::size_t n2d = static_cast<std::size_t>(batch) * features;
                std::size_t n1d = static_cast<std::size_t>(features);

                ensure_gpu_buffers(batch, features);

                cudaMemcpy(d_input_, input.data(), n2d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_gamma_, gamma_.data(), n1d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_beta_, beta_.data(), n1d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_running_mean_, running_mean_.data(), n1d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_running_var_, running_var_.data(), n1d * sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int smem = block * sizeof(float);

                Kernels::GPU::batch_norm_forward_kernel<<<features, block, smem>>>(
                    d_input_, d_output_, d_x_hat_,
                    d_gamma_, d_beta_,
                    d_batch_mean_, d_batch_var_,
                    d_running_mean_, d_running_var_,
                    batch, features,
                    eps_, momentum_, training_);

                cudaMemcpy(output.data(), d_output_, n2d * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(x_hat_.data(), d_x_hat_, n2d * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(batch_mean_.data(), d_batch_mean_, n1d * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(batch_var_.data(), d_batch_var_, n1d * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(running_mean_.data(), d_running_mean_, n1d * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(running_var_.data(), d_running_var_, n1d * sizeof(float), cudaMemcpyDeviceToHost);
            }

            void BatchNorm::backward_gpu(const Eigen::Tensor<float, 2>& grad_output,
                                         Eigen::Tensor<float, 2>& grad_input)
            {
                int batch = grad_output.dimension(0);
                int features = grad_output.dimension(1);
                std::size_t n2d = static_cast<std::size_t>(batch) * features;
                std::size_t n1d = static_cast<std::size_t>(features);

                cudaMemcpy(d_grad_output_, grad_output.data(), n2d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_input_, input_cache_.data(), n2d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_x_hat_, x_hat_.data(), n2d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_gamma_, gamma_.data(), n1d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_batch_mean_, batch_mean_.data(), n1d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_batch_var_, batch_var_.data(), n1d * sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int smem = 4 * block * sizeof(float);

                Kernels::GPU::batch_norm_backward_kernel<<<features, block, smem>>>(
                    d_grad_output_, d_input_, d_x_hat_,
                    d_gamma_, d_batch_mean_, d_batch_var_,
                    d_grad_input_, d_grad_gamma_, d_grad_beta_,
                    batch, features, eps_);

                cudaMemcpy(grad_input.data(), d_grad_input_, n2d * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(grad_gamma_.data(), d_grad_gamma_, n1d * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(grad_beta_.data(), d_grad_beta_, n1d * sizeof(float), cudaMemcpyDeviceToHost);
            }

        #endif

        Eigen::Tensor<float, 2> BatchNorm::forward(
            const Eigen::Tensor<float, 2>& input)
        {
            int batch = input.dimension(0);
            int features = input.dimension(1);

            if (features != num_features_)
                throw std::runtime_error("BatchNorm: input features (" +
                    std::to_string(features) + ") != num_features (" +
                    std::to_string(num_features_) + ")");

            input_cache_ = input;
            batch_size_cache_ = batch;

            Eigen::Tensor<float, 2> output(batch, features);
            x_hat_.resize(batch, features);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    forward_gpu(input, output);
                    return output;
                }
            #endif

            if (device_ == "cpu")
            {
                if (training_)
                {
                    batch_mean_.setZero();
                    for (int n = 0; n < batch; ++n)
                        for (int f = 0; f < features; ++f)
                            batch_mean_(f) += input(n, f);
                    float inv_batch = 1.0f / static_cast<float>(batch);
                    for (int f = 0; f < features; ++f)
                        batch_mean_(f) *= inv_batch;

                    batch_var_.setZero();
                    for (int n = 0; n < batch; ++n)
                        for (int f = 0; f < features; ++f)
                        {
                            float diff = input(n, f) - batch_mean_(f);
                            batch_var_(f) += diff * diff;
                        }
                    for (int f = 0; f < features; ++f)
                        batch_var_(f) *= inv_batch;

                    #pragma omp parallel for
                    for (int n = 0; n < batch; ++n)
                        for (int f = 0; f < features; ++f)
                        {
                            x_hat_(n, f) = (input(n, f) - batch_mean_(f)) /
                                           std::sqrt(batch_var_(f) + eps_);
                            output(n, f) = gamma_(f) * x_hat_(n, f) + beta_(f);
                        }

                    for (int f = 0; f < features; ++f)
                    {
                        running_mean_(f) = (1.0f - momentum_) * running_mean_(f) +
                                           momentum_ * batch_mean_(f);
                        running_var_(f) = (1.0f - momentum_) * running_var_(f) +
                                          momentum_ * batch_var_(f);
                    }
                }
                else
                {
                    #pragma omp parallel for
                    for (int n = 0; n < batch; ++n)
                        for (int f = 0; f < features; ++f)
                        {
                            x_hat_(n, f) = (input(n, f) - running_mean_(f)) /
                                           std::sqrt(running_var_(f) + eps_);
                            output(n, f) = gamma_(f) * x_hat_(n, f) + beta_(f);
                        }
                }
            }
            else // cpu-eigen
            {
                if (training_)
                {
                    batch_mean_.setZero();
                    for (int n = 0; n < batch; ++n)
                        for (int f = 0; f < features; ++f)
                            batch_mean_(f) += input(n, f);
                    float inv_batch = 1.0f / static_cast<float>(batch);
                    for (int f = 0; f < features; ++f)
                        batch_mean_(f) *= inv_batch;

                    batch_var_.setZero();
                    for (int n = 0; n < batch; ++n)
                        for (int f = 0; f < features; ++f)
                        {
                            float diff = input(n, f) - batch_mean_(f);
                            batch_var_(f) += diff * diff;
                        }
                    for (int f = 0; f < features; ++f)
                        batch_var_(f) *= inv_batch;

                    for (int n = 0; n < batch; ++n)
                        for (int f = 0; f < features; ++f)
                        {
                            x_hat_(n, f) = (input(n, f) - batch_mean_(f)) /
                                           std::sqrt(batch_var_(f) + eps_);
                            output(n, f) = gamma_(f) * x_hat_(n, f) + beta_(f);
                        }

                    for (int f = 0; f < features; ++f)
                    {
                        running_mean_(f) = (1.0f - momentum_) * running_mean_(f) +
                                           momentum_ * batch_mean_(f);
                        running_var_(f) = (1.0f - momentum_) * running_var_(f) +
                                          momentum_ * batch_var_(f);
                    }
                }
                else
                {
                    for (int n = 0; n < batch; ++n)
                        for (int f = 0; f < features; ++f)
                        {
                            x_hat_(n, f) = (input(n, f) - running_mean_(f)) /
                                           std::sqrt(running_var_(f) + eps_);
                            output(n, f) = gamma_(f) * x_hat_(n, f) + beta_(f);
                        }
                }
            }

            return output;
        }

        Eigen::Tensor<float, 2> BatchNorm::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            int batch = grad_output.dimension(0);
            int features = grad_output.dimension(1);
            float inv_batch = 1.0f / static_cast<float>(batch);

            Eigen::Tensor<float, 2> grad_input(batch, features);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    backward_gpu(grad_output, grad_input);
                    return grad_input;
                }
            #endif

            grad_gamma_.setZero();
            grad_beta_.setZero();

            for (int n = 0; n < batch; ++n)
                for (int f = 0; f < features; ++f)
                {
                    grad_gamma_(f) += grad_output(n, f) * x_hat_(n, f);
                    grad_beta_(f) += grad_output(n, f);
                }

            for (int f = 0; f < features; ++f)
            {
                float inv_std = 1.0f / std::sqrt(batch_var_(f) + eps_);

                float dvar = 0.0f;
                float dmean = 0.0f;

                for (int n = 0; n < batch; ++n)
                {
                    float dx_hat = grad_output(n, f) * gamma_(f);
                    float x_mu = input_cache_(n, f) - batch_mean_(f);

                    dvar += dx_hat * x_mu * (-0.5f) * inv_std * inv_std * inv_std;
                    dmean += dx_hat * (-inv_std);
                }

                for (int n = 0; n < batch; ++n)
                {
                    float dx_hat = grad_output(n, f) * gamma_(f);
                    float x_mu = input_cache_(n, f) - batch_mean_(f);

                    grad_input(n, f) = dx_hat * inv_std +
                                       dvar * 2.0f * x_mu * inv_batch +
                                       dmean * inv_batch;
                }
            }

            return grad_input;
        }

        void BatchNorm::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            if (!trainable_) return;

            optimizer.update(gamma_.data(), grad_gamma_.data(),
                             num_features_, learning_rate);
            optimizer.update(beta_.data(), grad_beta_.data(),
                             num_features_, learning_rate);

            grad_gamma_.setZero();
            grad_beta_.setZero();
        }
    }
}
