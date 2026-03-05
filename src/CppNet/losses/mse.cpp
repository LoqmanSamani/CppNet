/**
 * @file mse.cpp
 * @brief Mean Squared Error loss implementation
 *
 * Forward: L = mean((pred - target)^2)   [or sum/none]
 * Backward: dL/dpred = 2 * (pred - target) / N  [for mean reduction]
 */

#include "CppNet/losses/mse.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Losses
    {
        MSE::MSE(const std::string& reduction, const std::string& device)
            : reduction_(reduction), device_(device)
        {
        }

        MSE::~MSE()
        {
            #ifdef USE_CUDA
                release_gpu();
            #endif
        }

        #ifdef USE_CUDA

            void MSE::ensure_gpu(std::size_t n)
            {
                if (gpu_init_ && gpu_buf_ >= n) return;
                release_gpu();
                cudaMalloc(&d_pred_, n * sizeof(float));
                cudaMalloc(&d_target_, n * sizeof(float));
                cudaMalloc(&d_loss_, sizeof(float));
                cudaMalloc(&d_grad_, n * sizeof(float));
                gpu_buf_ = n;
                gpu_init_ = true;
            }

            void MSE::release_gpu()
            {
                if (d_pred_)   cudaFree(d_pred_);
                if (d_target_) cudaFree(d_target_);
                if (d_loss_)   cudaFree(d_loss_);
                if (d_grad_)   cudaFree(d_grad_);
                d_pred_ = nullptr; d_target_ = nullptr;
                d_loss_ = nullptr; d_grad_ = nullptr;
                gpu_buf_ = 0; gpu_init_ = false;
            }

            float MSE::forward_gpu(const Eigen::Tensor<float, 2>& predictions,
                                   const Eigen::Tensor<float, 2>& targets)
            {
                std::size_t n = predictions.size();
                ensure_gpu(n);

                cudaMemcpy(d_pred_, predictions.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_target_, targets.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                float zero = 0.0f;
                cudaMemcpy(d_loss_, &zero, sizeof(float), cudaMemcpyHostToDevice);

                int block = 256;
                int grid = (static_cast<int>(n) + block - 1) / block;
                Kernels::GPU::mse_forward_kernel<<<grid, block>>>(d_pred_, d_target_, d_loss_, static_cast<int>(n));

                float loss;
                cudaMemcpy(&loss, d_loss_, sizeof(float), cudaMemcpyDeviceToHost);

                if (reduction_ == "mean")
                    return loss / static_cast<float>(n);
                return loss;
            }

            void MSE::backward_gpu(const Eigen::Tensor<float, 2>& predictions,
                                   const Eigen::Tensor<float, 2>& targets,
                                   Eigen::Tensor<float, 2>& grad)
            {
                std::size_t n = predictions.size();
                ensure_gpu(n);

                cudaMemcpy(d_pred_, predictions.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_target_, targets.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                float scale = (reduction_ == "mean") ? 2.0f / static_cast<float>(n) : 2.0f;

                int block = 256;
                int grid = (static_cast<int>(n) + block - 1) / block;
                Kernels::GPU::mse_backward_kernel<<<grid, block>>>(d_pred_, d_target_, d_grad_, static_cast<int>(n), scale);

                cudaMemcpy(grad.data(), d_grad_, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

        #endif

        float MSE::forward(const Eigen::Tensor<float, 2>& predictions,
                           const Eigen::Tensor<float, 2>& targets)
        {
            int batch = predictions.dimension(0);
            int features = predictions.dimension(1);
            int total = batch * features;

            #ifdef USE_CUDA
                if (device_ == "gpu") return forward_gpu(predictions, targets);
            #endif

            float sum = 0.0f;

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2) reduction(+:sum)
            #endif
            for (int i = 0; i < batch; ++i)
            {
                for (int j = 0; j < features; ++j)
                {
                    float diff = predictions(i, j) - targets(i, j);
                    sum += diff * diff;
                }
            }

            if (reduction_ == "mean")
                return sum / static_cast<float>(total);
            else if (reduction_ == "sum")
                return sum;
            else
                return sum;
        }

        Eigen::Tensor<float, 2> MSE::backward(const Eigen::Tensor<float, 2>& predictions,
                                               const Eigen::Tensor<float, 2>& targets)
        {
            int batch = predictions.dimension(0);
            int features = predictions.dimension(1);
            int total = batch * features;

            Eigen::Tensor<float, 2> grad(batch, features);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    backward_gpu(predictions, targets, grad);
                    return grad;
                }
            #endif

            float scale = (reduction_ == "mean") ? 2.0f / static_cast<float>(total) : 2.0f;

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2)
            #endif
            for (int i = 0; i < batch; ++i)
                for (int j = 0; j < features; ++j)
                    grad(i, j) = scale * (predictions(i, j) - targets(i, j));

            return grad;
        }

        void MSE::set_num_threads(int num_threads)
        {
            #ifdef USE_OPENMP
            omp_set_num_threads(num_threads);
            #endif
            (void)num_threads;
        }
    }
}
