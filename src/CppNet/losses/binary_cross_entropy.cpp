#include <cmath>
#include <Eigen/Dense>
#include "CppNet/losses/binary_cross_entropy.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"
#ifdef USE_OPENMP
#include <omp.h>
#endif



namespace CppNet
{
    namespace Losses
    {
        BinaryCrossEntropy::BinaryCrossEntropy(const std::string& reduction, bool from_logits,
                                               float pos_weight, const std::string& device)
            : reduction_(reduction), from_logits_(from_logits), pos_weight_(pos_weight), device_(device)
        {
            if (reduction_ != "mean" && reduction_ != "sum" && reduction_ != "none")
            {
                throw std::runtime_error("Invalid reduction method: " + reduction_ + ". Must be 'mean', 'sum', or 'none'.");
            }
        }

        BinaryCrossEntropy::~BinaryCrossEntropy()
        {
            #ifdef USE_CUDA
                release_gpu();
            #endif
        }

        #ifdef USE_CUDA

            void BinaryCrossEntropy::ensure_gpu(std::size_t n)
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

            void BinaryCrossEntropy::release_gpu()
            {
                if (d_pred_)   cudaFree(d_pred_);
                if (d_target_) cudaFree(d_target_);
                if (d_loss_)   cudaFree(d_loss_);
                if (d_grad_)   cudaFree(d_grad_);
                d_pred_ = nullptr; d_target_ = nullptr;
                d_loss_ = nullptr; d_grad_ = nullptr;
                gpu_buf_ = 0; gpu_init_ = false;
            }

            float BinaryCrossEntropy::forward_gpu(const Eigen::Tensor<float, 2>& predictions,
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
                Kernels::GPU::bce_forward_kernel<<<grid, block>>>(d_pred_, d_target_, d_loss_, static_cast<int>(n));

                float loss;
                cudaMemcpy(&loss, d_loss_, sizeof(float), cudaMemcpyDeviceToHost);

                if (reduction_ == "mean" || reduction_ == "none")
                    return loss / static_cast<float>(n);
                return loss;
            }

            void BinaryCrossEntropy::backward_gpu(const Eigen::Tensor<float, 2>& predictions,
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
                Kernels::GPU::bce_backward_kernel<<<grid, block>>>(d_pred_, d_target_, d_grad_, static_cast<int>(n), scale);

                cudaMemcpy(grad.data(), d_grad_, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

        #endif
        
        void BinaryCrossEntropy::validate_inputs(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets)
        {
            if (predictions.dimension(0) != targets.dimension(0) || predictions.dimension(1) != targets.dimension(1))
            {
                throw std::runtime_error("Shape mismatch: predictions and targets must have the same dimensions!");
            }
            if (targets.size() == 0)
            {
                throw std::runtime_error("Empty input: predictions and targets cannot be empty!");
            }
            bool valid_labels = true;
            const int rows = targets.dimension(0);
            const int cols = targets.dimension(1);

            #pragma omp parallel for collapse(2) reduction(&&:valid_labels)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    float val = targets(i, j);
                    if (val != 0.0 && val != 1.0)
                    {
                       valid_labels = false;
                    }
                }
            }
            if (!valid_labels)
            {
                throw std::runtime_error("targets must contain binary labels (0 or 1)!");
            }
        }

        void BinaryCrossEntropy::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                #ifdef USE_OPENMP
                omp_set_num_threads(num_threads);
                #endif
            }
        }

        float BinaryCrossEntropy::forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets)
        {
            validate_inputs(predictions, targets);
            
            const int rows = predictions.dimension(0);
            const int cols = predictions.dimension(1);
            const int total_size = rows * cols;

            #ifdef USE_CUDA
                if (device_ == "gpu") return forward_gpu(predictions, targets);
            #endif

            float total_loss = 0.0f;
            
            #pragma omp parallel for collapse(2) reduction(+:total_loss)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    // clip values to [1e-7, 1 - 1e-7] to avoid log(0) and division by 0
                    float pred_clipped = std::max(1e-7f, std::min(predictions(i, j), 1.0f - 1e-7f));
                    float target_val = targets(i, j);
                    
                    // binary cross-entropy: -[y * log(y_hat) + (1-y) * log(1-y_hat)]
                    float loss_val = -(target_val * std::log(pred_clipped) + (1.0f - target_val) * std::log(1.0f - pred_clipped));
                    total_loss += loss_val;
                }
            }
            
            if (reduction_ == "mean")
            {
                return total_loss / total_size;
            }
            else if (reduction_ == "sum")
            {
                return total_loss;
            }
            else // "none"
            {
                return total_loss / total_size;
            }
        }

        Eigen::Tensor<float, 2> BinaryCrossEntropy::backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets)
        {
            validate_inputs(predictions, targets);
            const int rows = predictions.dimension(0);
            const int cols = predictions.dimension(1);
            const int total_size = rows * cols;
            
            Eigen::Tensor<float, 2> grad(rows, cols);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    backward_gpu(predictions, targets, grad);
                    return grad;
                }
            #endif
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    // clip values to [eps, 1-eps] to avoid log(0) and division by 0
                    float pred_clipped = std::max(1e-7f, std::min(predictions(i, j), 1.0f - 1e-7f));
                    float target_val = targets(i, j);
                    
                    // gradient: (y_hat - y) / (y_hat * (1 - y_hat))
                    float grad_val = (pred_clipped - target_val) / (pred_clipped * (1.0 - pred_clipped));
                    
                    if (reduction_ == "mean")
                    {
                        grad_val /= total_size;
                    }
                    
                    grad(i, j) = grad_val;
                }
            }
            
            return grad;
        }
    }
}