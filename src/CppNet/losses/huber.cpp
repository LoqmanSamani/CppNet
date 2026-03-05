/**
 * @file huber.cpp
 * @brief Huber (Smooth L1) loss implementation
 *
 * Forward: L = 0.5*(pred-target)^2         if |pred-target| <= delta
 *          L = delta*(|pred-target| - 0.5*delta) otherwise
 * Backward: dL/dpred = (pred-target)       if |pred-target| <= delta
 *           dL/dpred = delta*sign(pred-target) otherwise
 */

#include "CppNet/losses/huber.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Losses
    {
        Huber::Huber(float delta, const std::string& reduction, const std::string& device)
            : delta_(delta), reduction_(reduction), device_(device)
        {
        }

        Huber::~Huber()
        {
            #ifdef USE_CUDA
                release_gpu();
            #endif
        }

        #ifdef USE_CUDA

            void Huber::ensure_gpu(std::size_t n)
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

            void Huber::release_gpu()
            {
                if (d_pred_)   cudaFree(d_pred_);
                if (d_target_) cudaFree(d_target_);
                if (d_loss_)   cudaFree(d_loss_);
                if (d_grad_)   cudaFree(d_grad_);
                d_pred_ = nullptr; d_target_ = nullptr;
                d_loss_ = nullptr; d_grad_ = nullptr;
                gpu_buf_ = 0; gpu_init_ = false;
            }

            float Huber::forward_gpu(const Eigen::Tensor<float, 2>& predictions,
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
                Kernels::GPU::huber_forward_kernel<<<grid, block>>>(d_pred_, d_target_, d_loss_, static_cast<int>(n), delta_);

                float loss;
                cudaMemcpy(&loss, d_loss_, sizeof(float), cudaMemcpyDeviceToHost);

                if (reduction_ == "mean")
                    return loss / static_cast<float>(n);
                return loss;
            }

            void Huber::backward_gpu(const Eigen::Tensor<float, 2>& predictions,
                                     const Eigen::Tensor<float, 2>& targets,
                                     Eigen::Tensor<float, 2>& grad)
            {
                std::size_t n = predictions.size();
                ensure_gpu(n);

                cudaMemcpy(d_pred_, predictions.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_target_, targets.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                float scale = (reduction_ == "mean") ? 1.0f / static_cast<float>(n) : 1.0f;

                int block = 256;
                int grid = (static_cast<int>(n) + block - 1) / block;
                Kernels::GPU::huber_backward_kernel<<<grid, block>>>(d_pred_, d_target_, d_grad_, static_cast<int>(n), scale, delta_);

                cudaMemcpy(grad.data(), d_grad_, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

        #endif

        float Huber::forward(const Eigen::Tensor<float, 2>& predictions,
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
                    float abs_diff = std::fabs(diff);

                    if (abs_diff <= delta_)
                        sum += 0.5f * diff * diff;
                    else
                        sum += delta_ * (abs_diff - 0.5f * delta_);
                }
            }

            if (reduction_ == "mean")
                return sum / static_cast<float>(total);
            else if (reduction_ == "sum")
                return sum;
            else
                return sum;
        }

        Eigen::Tensor<float, 2> Huber::backward(const Eigen::Tensor<float, 2>& predictions,
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

            float scale = (reduction_ == "mean") ? 1.0f / static_cast<float>(total) : 1.0f;

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2)
            #endif
            for (int i = 0; i < batch; ++i)
                for (int j = 0; j < features; ++j)
                {
                    float diff = predictions(i, j) - targets(i, j);
                    float abs_diff = std::fabs(diff);

                    if (abs_diff <= delta_)
                        grad(i, j) = scale * diff;
                    else
                        grad(i, j) = scale * delta_ * ((diff > 0.0f) ? 1.0f : -1.0f);
                }

            return grad;
        }

        void Huber::set_num_threads(int num_threads)
        {
            #ifdef USE_OPENMP
            omp_set_num_threads(num_threads);
            #endif
            (void)num_threads;
        }
    }
}
