/**
 * @file linear.cpp
 * @brief Linear (fully-connected / dense) layer implementation
 *
 * Forward:   y = x * W + b
 * Backward:  dL/dx = dL/dy * W^T
 *            dL/dW += x^T * dL/dy
 *            dL/db += sum_batch(dL/dy)
 */

#include "CppNet/layers/linear.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/utils/init.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef USE_CUDA
#include "CppNet/kernels/gpu/gpu.hpp"
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Layers
    {
        // ────────────────────────────────────────────────────────────────
        //  Construction / destruction
        // ────────────────────────────────────────────────────────────────

        Linear::Linear(int in_size, int out_size,
                       std::string layer_name, bool trainable, bool bias,
                       std::string device, std::string weight_init)
            : in_size_(in_size), out_size_(out_size),
              layer_name_(std::move(layer_name)), trainable_(trainable),
              bias_flag_(bias), device_(std::move(device)),
              weight_init_(std::move(weight_init))
        {
            #ifdef USE_CUDA
            d_weights_      = nullptr;
            d_bias_         = nullptr;
            d_input_cache_  = nullptr;
            d_output_       = nullptr;
            d_grad_weights_ = nullptr;
            d_grad_bias_    = nullptr;
            d_grad_input_   = nullptr;
            d_grad_output_  = nullptr;
            gpu_initialized_   = false;
            gpu_max_batch_size_ = 0;
            #endif

            if (in_size <= 0 || out_size <= 0)
                throw std::runtime_error(
                    "in_size and out_size of layer: " + layer_name_ +
                    " must be positive integers!");

            if (layer_name_.empty())
                layer_name_ = "Linear_" + std::to_string(in_size_) + "x"
                            + std::to_string(out_size_);

            init_params_and_grads();
        }

        Linear::~Linear()
        {
            #ifdef USE_CUDA
            cleanup_gpu_buffers();
            #endif
        }

        void Linear::init_params_and_grads()
        {
            weights_      = CppNet::Utils::init_weights(in_size_, out_size_, weight_init_);
            grad_weights_ = CppNet::Utils::constant_init(in_size_, out_size_, 0.0f);

            if (bias_flag_)
            {
                biases_.resize(out_size_);
                biases_.setZero();
                grad_biases_.resize(out_size_);
                grad_biases_.setZero();
            }
            else
            {
                biases_      = Eigen::Tensor<float, 1>(0);
                grad_biases_ = Eigen::Tensor<float, 1>(0);
            }
        }

        void Linear::reinitialize_weights(const std::string& method)
        {
            std::string old = weight_init_;
            weight_init_ = method;
            try
            {
                init_params_and_grads();
                std::cout << "Layer '" << layer_name_
                          << "' weights reinitialized from '" << old
                          << "' to '" << method << "'" << std::endl;
            }
            catch (const std::exception& e)
            {
                weight_init_ = old;
                throw std::runtime_error(
                    "Failed to reinitialize with method '" + method + "': " + e.what());
            }
        }

        void Linear::reset_grads()
        {
            if (!trainable_) return;
            grad_weights_.setZero();
            if (bias_flag_ && grad_biases_.size() > 0)
                grad_biases_.setZero();
        }

        void Linear::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            if (!trainable_) return;

            optimizer.update(weights_.data(), grad_weights_.data(),
                             weights_.size(), learning_rate);

            if (bias_flag_)
                optimizer.update(biases_.data(), grad_biases_.data(),
                                 biases_.size(), learning_rate);
        }

        Eigen::Tensor<float, 2> Linear::forward(
            const Eigen::Tensor<float, 2>& input)
        {
            if (input.dimension(1) != weights_.dimension(0))
                throw std::runtime_error(
                    "Shape mismatch: in layer: " + layer_name_ +
                    " input.dimension(1) must be equal weights_.dimension(0)!");

            in_cache_ = input;

            const int batch_size  = input.dimension(0);
            const int input_size  = input.dimension(1);
            const int output_size = weights_.dimension(1);
            Eigen::Tensor<float, 2> output(batch_size, output_size);

            if (device_ == "cpu")
            {
                Eigen::Tensor<float, 2> out(batch_size, output_size);
                forward_cpu(input, out, batch_size, input_size, output_size);
                return out;
            }
            else if (device_ == "gpu")
            {
                #ifdef USE_CUDA
                if (gpu_output_cache_.dimension(0) < batch_size ||
                    gpu_output_cache_.dimension(1) != output_size)
                    gpu_output_cache_.resize(batch_size, output_size);

                forward_gpu(input, gpu_output_cache_,
                            batch_size, input_size, output_size);
                return gpu_output_cache_.slice(
                    Eigen::array<Eigen::Index, 2>{0, 0},
                    Eigen::array<Eigen::Index, 2>{batch_size, output_size});
                #else
                throw std::runtime_error("GPU device selected but CUDA not available");
                #endif
            }
            else if (device_ == "cpu-eigen")
            {
                forward_eigen(input, output, batch_size, input_size, output_size);
                return output;
            }

            throw std::runtime_error("Unknown device: " + device_);
        }

        void Linear::forward_cpu(const Eigen::Tensor<float, 2>& input,
                                 Eigen::Tensor<float, 2>& output,
                                 int batch_size, int input_size, int output_size)
        {
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < batch_size; ++b)
                for (int j = 0; j < output_size; ++j)
                {
                    float sum = 0.0f;
                    #pragma omp simd reduction(+:sum)
                    for (int i = 0; i < input_size; ++i)
                        sum += input(b, i) * weights_(i, j);
                    output(b, j) = sum;
                }

            if (bias_flag_)
            {
                #pragma omp parallel for collapse(2)
                for (int b = 0; b < batch_size; ++b)
                    for (int j = 0; j < output_size; ++j)
                        output(b, j) += biases_(j);
            }
        }

        void Linear::forward_eigen(const Eigen::Tensor<float, 2>& input,
                                   Eigen::Tensor<float, 2>& output,
                                   int batch_size, int input_size, int output_size)
        {
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
                {Eigen::IndexPair<int>(1, 0)};
            output = input.contract(weights_, product_dims);

            if (bias_flag_)
            {
                Eigen::array<Eigen::Index, 2> bcast({input.dimension(0), 1});
                Eigen::Tensor<float, 2> bias_2d = biases_.reshape(
                    Eigen::array<Eigen::Index, 2>({1, biases_.dimension(0)}))
                    .broadcast(bcast);
                output = output + bias_2d;
            }
        }

     
        Eigen::Tensor<float, 2> Linear::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            const int batch_size  = grad_output.dimension(0);
            const int output_size = grad_output.dimension(1);
            const int input_size  = in_cache_.dimension(1);

            if (grad_output.dimension(1) != out_size_)
                throw std::runtime_error(
                    "Output size mismatch in layer: " + layer_name_);

            if (device_ == "cpu")
            {
                Eigen::Tensor<float, 2> grad_input(batch_size, input_size);
                grad_input.setZero();
                backward_cpu(grad_output, grad_input,
                             batch_size, output_size, input_size);
                return grad_input;
            }
            else if (device_ == "gpu")
            {
                #ifdef USE_CUDA
                if (gpu_grad_input_cache_.dimension(0) < batch_size ||
                    gpu_grad_input_cache_.dimension(1) != input_size)
                    gpu_grad_input_cache_.resize(batch_size, input_size);

                cudaMemcpy(d_grad_output_, grad_output.data(),
                           batch_size * output_size * sizeof(float),
                           cudaMemcpyHostToDevice);

                backward_gpu(batch_size, output_size, input_size);

                cudaMemcpy(gpu_grad_input_cache_.data(), d_grad_input_,
                           batch_size * input_size * sizeof(float),
                           cudaMemcpyDeviceToHost);
                return gpu_grad_input_cache_;
                #else
                throw std::runtime_error("GPU device selected but CUDA not available");
                #endif
            }
            else if (device_ == "cpu-eigen")
            {
                Eigen::Tensor<float, 2> grad_input(batch_size, input_size);
                grad_input.setZero();
                backward_eigen(grad_output, grad_input,
                               batch_size, output_size, input_size);
                return grad_input;
            }

            throw std::runtime_error("Unknown device: " + device_);
        }

        void Linear::backward_cpu(const Eigen::Tensor<float, 2>& grad_output,
                                  Eigen::Tensor<float, 2>& grad_input,
                                  int batch_size, int output_size, int input_size)
        {
            if (trainable_)
            {
                // dL/dW += X^T * dL/dY
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < input_size; ++i)
                    for (int j = 0; j < output_size; ++j)
                    {
                        float sum = 0.0f;
                        #pragma omp simd reduction(+:sum)
                        for (int b = 0; b < batch_size; ++b)
                            sum += in_cache_(b, i) * grad_output(b, j);
                        grad_weights_(i, j) += sum;
                    }

                // dL/db += sum_batch(dL/dY)
                if (bias_flag_)
                {
                    #pragma omp parallel for
                    for (int j = 0; j < output_size; ++j)
                    {
                        float sum = 0.0f;
                        #pragma omp simd reduction(+:sum)
                        for (int b = 0; b < batch_size; ++b)
                            sum += grad_output(b, j);
                        grad_biases_(j) += sum;
                    }
                }
            }

            // dL/dX = dL/dY * W^T
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < batch_size; ++b)
                for (int i = 0; i < input_size; ++i)
                {
                    float sum = 0.0f;
                    #pragma omp simd reduction(+:sum)
                    for (int j = 0; j < output_size; ++j)
                        sum += grad_output(b, j) * weights_(i, j);
                    grad_input(b, i) += sum;
                }
        }

        void Linear::backward_eigen(const Eigen::Tensor<float, 2>& grad_output,
                                    Eigen::Tensor<float, 2>& grad_input,
                                    int batch_size, int output_size, int input_size)
        {
            Eigen::array<int, 2> transpose({1, 0});
            Eigen::array<Eigen::IndexPair<int>, 1> contract =
                {Eigen::IndexPair<int>(1, 0)};

            if (trainable_)
            {
                // dL/dW += X^T * dL/dY
                grad_weights_ += in_cache_.shuffle(transpose)
                                          .contract(grad_output, contract);
                // dL/db += sum_batch(dL/dY)
                if (bias_flag_)
                {
                    Eigen::array<int, 1> batch_dim({0});
                    grad_biases_ += grad_output.sum(batch_dim);
                }
            }

            // dL/dX = dL/dY * W^T
            grad_input += grad_output.contract(weights_.shuffle(transpose), contract);
        }

        #ifdef USE_CUDA

        void Linear::set_max_batch_size(int max_batch_size)
        {
            if (device_ != "gpu") return;
            init_gpu_buffers(max_batch_size);
            gpu_output_cache_.resize(max_batch_size, out_size_);
            gpu_grad_input_cache_.resize(max_batch_size, in_size_);
        }

        void Linear::init_gpu_buffers(int max_batch_size)
        {
            if (gpu_initialized_) cleanup_gpu_buffers();
            gpu_max_batch_size_ = max_batch_size;

            std::cout << "Allocating GPU buffers for layer '"
                      << layer_name_ << "' (max batch: "
                      << max_batch_size << ")" << std::endl;

            cudaMalloc(&d_weights_, in_size_ * out_size_ * sizeof(float));
            if (bias_flag_)
            {
                cudaMalloc(&d_bias_,      out_size_ * sizeof(float));
                cudaMalloc(&d_grad_bias_, out_size_ * sizeof(float));
            }
            cudaMalloc(&d_input_cache_,  max_batch_size * in_size_  * sizeof(float));
            cudaMalloc(&d_output_,       max_batch_size * out_size_ * sizeof(float));
            cudaMalloc(&d_grad_weights_, in_size_ * out_size_ * sizeof(float));
            cudaMalloc(&d_grad_input_,   max_batch_size * in_size_  * sizeof(float));
            cudaMalloc(&d_grad_output_,  max_batch_size * out_size_ * sizeof(float));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                throw std::runtime_error(
                    "CUDA allocation failed for layer '" + layer_name_ +
                    "': " + cudaGetErrorString(err));

            sync_weights_to_gpu();
            gpu_initialized_ = true;

            std::cout << "GPU buffers allocated successfully for layer '"
                      << layer_name_ << "'" << std::endl;
        }

        void Linear::cleanup_gpu_buffers()
        {
            if (!gpu_initialized_) return;
            if (d_weights_)      cudaFree(d_weights_);
            if (d_bias_)         cudaFree(d_bias_);
            if (d_input_cache_)  cudaFree(d_input_cache_);
            if (d_output_)       cudaFree(d_output_);
            if (d_grad_weights_) cudaFree(d_grad_weights_);
            if (d_grad_bias_)    cudaFree(d_grad_bias_);
            if (d_grad_input_)   cudaFree(d_grad_input_);
            if (d_grad_output_)  cudaFree(d_grad_output_);
            d_weights_ = d_bias_ = d_input_cache_ = d_output_ = nullptr;
            d_grad_weights_ = d_grad_bias_ = d_grad_input_ = d_grad_output_ = nullptr;
            gpu_initialized_ = false;
        }

        void Linear::sync_weights_to_gpu()
        {
            if (!gpu_initialized_) return;
            cudaMemcpy(d_weights_, weights_.data(),
                       in_size_ * out_size_ * sizeof(float),
                       cudaMemcpyHostToDevice);
            if (bias_flag_)
                cudaMemcpy(d_bias_, biases_.data(),
                           out_size_ * sizeof(float),
                           cudaMemcpyHostToDevice);
        }

        void Linear::sync_weights_from_gpu()
        {
            if (!gpu_initialized_) return;
            cudaMemcpy(weights_.data(), d_weights_,
                       in_size_ * out_size_ * sizeof(float),
                       cudaMemcpyDeviceToHost);
            if (bias_flag_)
                cudaMemcpy(biases_.data(), d_bias_,
                           out_size_ * sizeof(float),
                           cudaMemcpyDeviceToHost);
        }

        void Linear::sync_gradients_from_gpu()
        {
            if (!gpu_initialized_ || !trainable_) return;
            cudaMemcpy(grad_weights_.data(), d_grad_weights_,
                       in_size_ * out_size_ * sizeof(float),
                       cudaMemcpyDeviceToHost);
            if (bias_flag_)
                cudaMemcpy(grad_biases_.data(), d_grad_bias_,
                           out_size_ * sizeof(float),
                           cudaMemcpyDeviceToHost);
        }

        void Linear::forward_gpu(const Eigen::Tensor<float, 2>& input,
                                 Eigen::Tensor<float, 2>& output,
                                 int batch_size, int input_size, int output_size)
        {
            if (!gpu_initialized_)
                throw std::runtime_error(
                    "GPU buffers not initialized for layer '" + layer_name_ +
                    "'. Call set_max_batch_size() first.");
            if (batch_size > gpu_max_batch_size_)
                throw std::runtime_error(
                    "Batch size " + std::to_string(batch_size) +
                    " exceeds GPU buffer size " + std::to_string(gpu_max_batch_size_) +
                    " in layer '" + layer_name_ + "'");

            cudaMemcpy(d_input_cache_, input.data(),
                       batch_size * input_size * sizeof(float),
                       cudaMemcpyHostToDevice);

            dim3 block(32, 32);
            dim3 grid((output_size + 31) / 32, (batch_size + 31) / 32);
            CppNet::Kernels::GPU::matmul_kernel<<<grid, block>>>(
                d_input_cache_, d_weights_, d_output_,
                batch_size, output_size, input_size);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                throw std::runtime_error(
                    "CUDA matmul kernel error: " + std::string(cudaGetErrorString(err)));

            if (bias_flag_)
            {
                dim3 bblock(16, 16);
                dim3 bgrid((output_size + 15) / 16, (batch_size + 15) / 16);
                CppNet::Kernels::GPU::add_bias_kernel<<<bgrid, bblock>>>(
                    d_output_, d_bias_, batch_size, output_size);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                    throw std::runtime_error(
                        "CUDA bias kernel error: " + std::string(cudaGetErrorString(err)));
            }

            cudaMemcpy(output.data(), d_output_,
                       batch_size * output_size * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }

        void Linear::backward_gpu(int batch_size, int output_size, int input_size)
        {
            if (trainable_)
            {
                cudaMemset(d_grad_weights_, 0,
                           input_size * output_size * sizeof(float));
                if (bias_flag_)
                    cudaMemset(d_grad_bias_, 0, output_size * sizeof(float));

                dim3 block(32, 32);
                dim3 grid_w((output_size + 31) / 32, (input_size + 31) / 32);
                Kernels::GPU::matmul_grad_weights_kernel<<<grid_w, block>>>(
                    d_input_cache_, d_grad_output_, d_grad_weights_,
                    batch_size, input_size, output_size);

                if (bias_flag_)
                {
                    dim3 bblock(256);
                    dim3 bgrid(output_size);
                    Kernels::GPU::bias_grad_kernel<<<bgrid, bblock>>>(
                        d_grad_output_, d_grad_bias_, batch_size, output_size);
                }
            }

            cudaMemset(d_grad_input_, 0,
                       batch_size * input_size * sizeof(float));
            dim3 block(32, 32);
            dim3 grid_x((input_size + 31) / 32, (batch_size + 31) / 32);
            Kernels::GPU::matmul_grad_input_kernel<<<grid_x, block>>>(
                d_grad_output_, d_weights_, d_grad_input_,
                batch_size, input_size, output_size);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                throw std::runtime_error(
                    "CUDA backward kernel error: " +
                    std::string(cudaGetErrorString(err)));
        }

        #endif // USE_CUDA

        void Linear::print_layer_info() const
        {
            std::cout << "  Layer: " << layer_name_ << '\n'
                      << "  Input size: " << in_size_ << '\n'
                      << "  Output size: " << out_size_ << '\n'
                      << "  Trainable: " << (trainable_ ? "Yes" : "No") << '\n'
                      << "  Has bias: " << (bias_flag_ ? "Yes" : "No") << '\n'
                      << "  Weight shape: [" << weights_.dimension(0) << ", "
                      << weights_.dimension(1) << "]" << '\n';
            if (bias_flag_)
                std::cout << "  Bias shape: [" << biases_.dimension(0) << "]"
                          << '\n';
        }

        void Linear::set_num_threads(int num_threads)
        {
            #ifdef USE_OPENMP
            if (num_threads > 0)
                omp_set_num_threads(num_threads);
            #endif
        }

    } // namespace Layers
} // namespace CppNet
