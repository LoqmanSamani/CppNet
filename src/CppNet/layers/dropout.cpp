/**
 * @file dropout.cpp
 * @brief Dropout layer implementation (inverted dropout)
 */

#include "CppNet/layers/dropout.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"
#include <chrono>
#include <stdexcept>

namespace CppNet
{
    namespace Layers
    {
        Dropout::Dropout(float p, const std::string& device)
            : p_(p),
              scale_(1.0f / (1.0f - p)),
              device_(device),
              gen_(static_cast<unsigned>(
                  std::chrono::steady_clock::now().time_since_epoch().count())),
              dist_(1.0 - static_cast<double>(p))
        {
            if (p_ < 0.0f || p_ >= 1.0f)
                throw std::invalid_argument("Dropout probability must be in [0, 1)");

            #ifdef USE_CUDA
                d_input_ = nullptr;
                d_mask_ = nullptr;
                d_output_ = nullptr;
                gpu_buffer_size_ = 0;
                gpu_initialized_ = false;
            #endif
        }

        Dropout::~Dropout()
        {
            #ifdef USE_CUDA
                release_gpu_buffers();
            #endif
        }

        #ifdef USE_CUDA

            void Dropout::ensure_gpu_buffer(std::size_t num_elements)
            {
                if (gpu_initialized_ && gpu_buffer_size_ >= num_elements)
                    return;

                release_gpu_buffers();

                cudaMalloc(&d_input_, num_elements * sizeof(float));
                cudaMalloc(&d_mask_, num_elements * sizeof(float));
                cudaMalloc(&d_output_, num_elements * sizeof(float));

                gpu_buffer_size_ = num_elements;
                gpu_initialized_ = true;
            }

            void Dropout::release_gpu_buffers()
            {
                if (d_input_) cudaFree(d_input_);
                if (d_mask_) cudaFree(d_mask_);
                if (d_output_) cudaFree(d_output_);

                d_input_ = nullptr;
                d_mask_ = nullptr;
                d_output_ = nullptr;
                gpu_buffer_size_ = 0;
                gpu_initialized_ = false;
            }

            void Dropout::forward_gpu_2d(const Eigen::Tensor<float, 2>& input,
                                         Eigen::Tensor<float, 2>& output)
            {
                std::size_t n = input.size();
                ensure_gpu_buffer(n);

                cudaMemcpy(d_input_, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_mask_, mask_2d_.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int grid = (static_cast<int>(n) + block - 1) / block;
                Kernels::GPU::dropout_forward_kernel<<<grid, block>>>(
                    d_input_, d_mask_, d_output_, static_cast<int>(n), scale_);

                cudaMemcpy(output.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

            void Dropout::backward_gpu_2d(const Eigen::Tensor<float, 2>& grad,
                                          Eigen::Tensor<float, 2>& grad_input)
            {
                std::size_t n = grad.size();

                cudaMemcpy(d_input_, grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_mask_, mask_2d_.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int grid = (static_cast<int>(n) + block - 1) / block;
                Kernels::GPU::dropout_backward_kernel<<<grid, block>>>(
                    d_input_, d_mask_, d_output_, static_cast<int>(n), scale_);

                cudaMemcpy(grad_input.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

            void Dropout::forward_gpu_4d(const Eigen::Tensor<float, 4>& input,
                                         Eigen::Tensor<float, 4>& output)
            {
                std::size_t n = input.size();
                ensure_gpu_buffer(n);

                cudaMemcpy(d_input_, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_mask_, mask_4d_.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int grid = (static_cast<int>(n) + block - 1) / block;
                Kernels::GPU::dropout_forward_kernel<<<grid, block>>>(
                    d_input_, d_mask_, d_output_, static_cast<int>(n), scale_);

                cudaMemcpy(output.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

            void Dropout::backward_gpu_4d(const Eigen::Tensor<float, 4>& grad,
                                          Eigen::Tensor<float, 4>& grad_input)
            {
                std::size_t n = grad.size();

                cudaMemcpy(d_input_, grad.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_mask_, mask_4d_.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int grid = (static_cast<int>(n) + block - 1) / block;
                Kernels::GPU::dropout_backward_kernel<<<grid, block>>>(
                    d_input_, d_mask_, d_output_, static_cast<int>(n), scale_);

                cudaMemcpy(grad_input.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

        #endif

        Eigen::Tensor<float, 2> Dropout::forward(
            const Eigen::Tensor<float, 2>& input)
        {
            if (!training_ || p_ == 0.0f)
                return input;

            int rows = input.dimension(0);
            int cols = input.dimension(1);

            mask_2d_.resize(rows, cols);
            Eigen::Tensor<float, 2> output(rows, cols);

            // Generate mask on CPU (stochastic operation)
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    mask_2d_(i, j) = dist_(gen_) ? 1.0f : 0.0f;

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    forward_gpu_2d(input, output);
                    return output;
                }
            #endif

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    output(i, j) = input(i, j) * mask_2d_(i, j) * scale_;

            return output;
        }

        Eigen::Tensor<float, 2> Dropout::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            if (!training_ || p_ == 0.0f)
                return grad_output;

            int rows = grad_output.dimension(0);
            int cols = grad_output.dimension(1);
            Eigen::Tensor<float, 2> grad_input(rows, cols);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    backward_gpu_2d(grad_output, grad_input);
                    return grad_input;
                }
            #endif

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    grad_input(i, j) = grad_output(i, j) * mask_2d_(i, j) * scale_;

            return grad_input;
        }

        Eigen::Tensor<float, 4> Dropout::forward(
            const Eigen::Tensor<float, 4>& input)
        {
            if (!training_ || p_ == 0.0f)
                return input;

            int d0 = input.dimension(0);
            int d1 = input.dimension(1);
            int d2 = input.dimension(2);
            int d3 = input.dimension(3);

            mask_4d_.resize(d0, d1, d2, d3);
            Eigen::Tensor<float, 4> output(d0, d1, d2, d3);

            // Generate mask on CPU (stochastic operation)
            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                        for (int d = 0; d < d3; ++d)
                            mask_4d_(a, b, c, d) = dist_(gen_) ? 1.0f : 0.0f;

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    forward_gpu_4d(input, output);
                    return output;
                }
            #endif

            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                        for (int d = 0; d < d3; ++d)
                            output(a, b, c, d) = input(a, b, c, d) * mask_4d_(a, b, c, d) * scale_;

            return output;
        }

        Eigen::Tensor<float, 4> Dropout::backward(
            const Eigen::Tensor<float, 4>& grad_output)
        {
            if (!training_ || p_ == 0.0f)
                return grad_output;

            int d0 = grad_output.dimension(0);
            int d1 = grad_output.dimension(1);
            int d2 = grad_output.dimension(2);
            int d3 = grad_output.dimension(3);

            Eigen::Tensor<float, 4> grad_input(d0, d1, d2, d3);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    backward_gpu_4d(grad_output, grad_input);
                    return grad_input;
                }
            #endif

            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                        for (int d = 0; d < d3; ++d)
                            grad_input(a, b, c, d) =
                                grad_output(a, b, c, d) * mask_4d_(a, b, c, d) * scale_;

            return grad_input;
        }

        void Dropout::step(Optimizers::Optimizer& /*optimizer*/, float /*learning_rate*/)
        {
            // No parameters to update
        }
    }
}
