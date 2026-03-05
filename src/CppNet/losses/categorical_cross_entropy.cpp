#include <cmath>
#include <Eigen/Dense>
#include "CppNet/losses/categorical_cross_entropy.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"
#ifdef USE_OPENMP
#include <omp.h>
#endif



namespace CppNet
{
    namespace Losses
    {
        CategoricalCrossEntropy::CategoricalCrossEntropy(const std::string& reduction, bool from_logits,
                                                         float label_smoothing, const std::string& device)
            : reduction_(reduction), from_logits_(from_logits), label_smoothing_(label_smoothing), device_(device) {}

        CategoricalCrossEntropy::~CategoricalCrossEntropy()
        {
            #ifdef USE_CUDA
                release_gpu();
            #endif
        }

        #ifdef USE_CUDA

            void CategoricalCrossEntropy::ensure_gpu(std::size_t n)
            {
                if (gpu_init_ && gpu_buf_ >= n) return;
                release_gpu();
                cudaMalloc(&d_pred_, n * sizeof(float));
                cudaMalloc(&d_target_, n * sizeof(float));
                cudaMalloc(&d_softmax_, n * sizeof(float));
                cudaMalloc(&d_loss_, sizeof(float));
                cudaMalloc(&d_grad_, n * sizeof(float));
                gpu_buf_ = n;
                gpu_init_ = true;
            }

            void CategoricalCrossEntropy::release_gpu()
            {
                if (d_pred_)    cudaFree(d_pred_);
                if (d_target_)  cudaFree(d_target_);
                if (d_softmax_) cudaFree(d_softmax_);
                if (d_loss_)    cudaFree(d_loss_);
                if (d_grad_)    cudaFree(d_grad_);
                d_pred_ = nullptr; d_target_ = nullptr;
                d_softmax_ = nullptr; d_loss_ = nullptr; d_grad_ = nullptr;
                gpu_buf_ = 0; gpu_init_ = false;
            }

            float CategoricalCrossEntropy::forward_gpu(const Eigen::Tensor<float, 2>& predictions,
                                                       const Eigen::Tensor<float, 2>& targets)
            {
                int batch_size = predictions.dimension(0);
                int num_classes = predictions.dimension(1);
                std::size_t n = predictions.size();
                ensure_gpu(n);

                // upload processed targets (label smoothing already applied by caller)
                cudaMemcpy(d_target_, targets_cache_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                float zero = 0.0f;
                cudaMemcpy(d_loss_, &zero, sizeof(float), cudaMemcpyHostToDevice);

                if (from_logits_)
                {
                    cudaMemcpy(d_pred_, predictions.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                    int block = 256;
                    int smem = block * sizeof(float);
                    Kernels::GPU::categorical_ce_logits_forward_kernel<<<batch_size, block, smem>>>(
                        d_pred_, d_target_, d_softmax_, d_loss_, batch_size, num_classes);

                    // download softmax for backward
                    softmax_cache_.resize(batch_size, num_classes);
                    cudaMemcpy(softmax_cache_.data(), d_softmax_, n * sizeof(float), cudaMemcpyDeviceToHost);
                }
                else
                {
                    // predictions are probabilities, copy them as softmax_cache
                    softmax_cache_ = predictions;
                    cudaMemcpy(d_pred_, predictions.data(), n * sizeof(float), cudaMemcpyHostToDevice);

                    int block = 256;
                    int grid = (static_cast<int>(n) + block - 1) / block;
                    Kernels::GPU::categorical_ce_probs_forward_kernel<<<grid, block>>>(
                        d_pred_, d_target_, d_loss_, static_cast<int>(n));
                }

                float loss;
                cudaMemcpy(&loss, d_loss_, sizeof(float), cudaMemcpyDeviceToHost);

                if (reduction_ == "mean")
                    return loss / static_cast<float>(batch_size);
                return loss;
            }

            void CategoricalCrossEntropy::backward_gpu(const Eigen::Tensor<float, 2>& predictions,
                                                       const Eigen::Tensor<float, 2>& targets,
                                                       Eigen::Tensor<float, 2>& grad)
            {
                int batch_size = predictions.dimension(0);
                int num_classes = predictions.dimension(1);
                std::size_t n = predictions.size();

                float scale = (reduction_ == "mean") ? 1.0f / static_cast<float>(batch_size) : 1.0f;

                int block = 256;
                int grid = (static_cast<int>(n) + block - 1) / block;

                if (from_logits_)
                {
                    cudaMemcpy(d_softmax_, softmax_cache_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_target_, targets_cache_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                    Kernels::GPU::categorical_ce_logits_backward_kernel<<<grid, block>>>(
                        d_softmax_, d_target_, d_grad_, static_cast<int>(n), scale);
                }
                else
                {
                    cudaMemcpy(d_pred_, softmax_cache_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_target_, targets_cache_.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                    Kernels::GPU::categorical_ce_probs_backward_kernel<<<grid, block>>>(
                        d_pred_, d_target_, d_grad_, static_cast<int>(n), scale);
                }

                cudaMemcpy(grad.data(), d_grad_, n * sizeof(float), cudaMemcpyDeviceToHost);
            }

        #endif

        void CategoricalCrossEntropy::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                #ifdef USE_OPENMP
                omp_set_num_threads(num_threads);
                #endif
            }
        }

        float CategoricalCrossEntropy::forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets) 
        {
            if (predictions.size() == 0 || targets.size() == 0)
                throw std::runtime_error("CategoricalCrossEntropy: Empty input tensors");

            int batch_size = predictions.dimension(0);
            int num_classes = predictions.dimension(1);

            if (targets.dimension(0) != batch_size)
                throw std::runtime_error("CategoricalCrossEntropy: Batch size mismatch between predictions and targets");

            Eigen::Tensor<float, 2> one_hot(batch_size, num_classes);
            one_hot.setZero();
            for (int b = 0; b < batch_size; ++b)
            {
                int cls = targets(b);
                if (cls < 0 || cls >= num_classes)
                    throw std::runtime_error("CategoricalCrossEntropy: Target index out of range");
                one_hot(b, cls) = 1.0f;
            }

            return forward(predictions, one_hot);
        }



        float CategoricalCrossEntropy::forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets)
        {
            if (predictions.size() == 0 || targets.size() == 0)
            {
                throw std::runtime_error("CategoricalCrossEntropy: Empty input tensors");
            }
            
            if (predictions.dimension(0) != targets.dimension(0) || 
                predictions.dimension(1) != targets.dimension(1))
            {
                throw std::runtime_error("CategoricalCrossEntropy: Predictions and targets shape mismatch");
            }
            
            int batch_size = predictions.dimension(0);
            int num_classes = predictions.dimension(1);
            
            targets_cache_ = targets;
            if (label_smoothing_ > 0.0)
            {
                float smoothing_factor = label_smoothing_ / num_classes;
                targets_cache_ = targets * (1.0f - label_smoothing_) + smoothing_factor;
            }

            #ifdef USE_CUDA
                if (device_ == "gpu") return forward_gpu(predictions, targets);
            #endif
            
            if (from_logits_)
            {
                softmax_cache_ = Eigen::Tensor<float, 2>(batch_size, num_classes);
                
                #pragma omp parallel for schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    float max_val = predictions(b, 0);
                    for (int j = 1; j < num_classes; ++j)
                    {
                        if (predictions(b, j) > max_val)
                            max_val = predictions(b, j);
                    }
                    
                    float sum_exp = 0.0f;
                    for (int j = 0; j < num_classes; ++j)
                    {
                        float exp_val = std::exp(predictions(b, j) - max_val);
                        softmax_cache_(b, j) = exp_val;
                        sum_exp += exp_val;
                    }
                    
                    for (int j = 0; j < num_classes; ++j)
                    {
                        softmax_cache_(b, j) /= sum_exp;
                    }
                }
            }
            else
            {
                softmax_cache_ = predictions;
            }
            
            float total_loss = 0.0f;
            const float epsilon = 1e-15f; // small value to prevent log(0)
            
            #pragma omp parallel for reduction(+:total_loss) schedule(static)
            for (int b = 0; b < batch_size; ++b)
            {
                float sample_loss = 0.0f;
                for (int j = 0; j < num_classes; ++j)
                {
                    float clipped_pred = std::max(epsilon, std::min(1.0f - epsilon, softmax_cache_(b, j)));
                    sample_loss -= targets_cache_(b, j) * std::log(clipped_pred);
                }
                total_loss += sample_loss;
            }
            
            // apply reduction
            if (reduction_ == "mean")
            {
                return total_loss / batch_size;
            }
            else if (reduction_ == "sum")
            {
                return total_loss;
            }
            else if (reduction_ == "none")
            {
                return total_loss;
            }
            else
            {
                throw std::runtime_error("CategoricalCrossEntropy: Invalid reduction type. Use 'mean', 'sum', or 'none'");
            }
        }

        float CategoricalCrossEntropy::forward(const Eigen::Tensor<float, 3>& predictions, const Eigen::Tensor<int, 2>& targets) 
        {
            if (predictions.size() == 0 || targets.size() == 0)
                throw std::runtime_error("CategoricalCrossEntropy: Empty input tensors");

            int batch_size = predictions.dimension(0);
            int seq_len = predictions.dimension(1);
            int num_classes = predictions.dimension(2);

            if (targets.dimension(0) != batch_size || targets.dimension(1) != seq_len)
                throw std::runtime_error("CategoricalCrossEntropy: Shape mismatch between predictions and targets");

            int total = batch_size * seq_len;
            Eigen::Tensor<float, 2> pred_2d(total, num_classes);
            Eigen::Tensor<float, 2> one_hot(total, num_classes);
            one_hot.setZero();

            for (int b = 0; b < batch_size; ++b)
            {
                for (int t = 0; t < seq_len; ++t)
                {
                    int idx = b * seq_len + t;
                    int cls = targets(b, t);
                    if (cls < 0 || cls >= num_classes)
                        throw std::runtime_error("CategoricalCrossEntropy: Target index out of range");
                    one_hot(idx, cls) = 1.0f;
                    for (int c = 0; c < num_classes; ++c)
                        pred_2d(idx, c) = predictions(b, t, c);
                }
            }

            return forward(pred_2d, one_hot);
        }

        Eigen::Tensor<float, 2> CategoricalCrossEntropy::backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets) 
        {
            int batch_size = predictions.dimension(0);
            int num_classes = predictions.dimension(1);

            if (targets.dimension(0) != batch_size)
                throw std::runtime_error("CategoricalCrossEntropy: Batch size mismatch in backward");

            Eigen::Tensor<float, 2> one_hot(batch_size, num_classes);
            one_hot.setZero();
            for (int b = 0; b < batch_size; ++b)
            {
                int cls = targets(b);
                if (cls < 0 || cls >= num_classes)
                    throw std::runtime_error("CategoricalCrossEntropy: Target index out of range");
                one_hot(b, cls) = 1.0f;
            }

            return backward(predictions, one_hot);
        }

        Eigen::Tensor<float, 2> CategoricalCrossEntropy::backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets)
        {
            if (softmax_cache_.size() == 0 || targets_cache_.size() == 0)
            {
                throw std::runtime_error("CategoricalCrossEntropy: Must call forward() before backward()");
            }
            
            if (predictions.dimension(0) != targets.dimension(0) || 
                predictions.dimension(1) != targets.dimension(1))
            {
                throw std::runtime_error("CategoricalCrossEntropy: Predictions and targets shape mismatch in backward");
            }
            
            int batch_size = predictions.dimension(0);
            int num_classes = predictions.dimension(1);
            
            Eigen::Tensor<float, 2> gradients(batch_size, num_classes);

            #ifdef USE_CUDA
                if (device_ == "gpu")
                {
                    backward_gpu(predictions, targets, gradients);
                    return gradients;
                }
            #endif
            
            if (from_logits_)
            {
                #pragma omp parallel for collapse(2) schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    for (int j = 0; j < num_classes; ++j)
                    {
                        gradients(b, j) = softmax_cache_(b, j) - targets_cache_(b, j);
                    }
                }
            }
            else
            {
                const float epsilon = 1e-15f;
                
                #pragma omp parallel for collapse(2) schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    for (int j = 0; j < num_classes; ++j)
                    {
                        float clipped_pred = std::max(epsilon, std::min(1.0f - epsilon, softmax_cache_(b, j)));
                        gradients(b, j) = -targets_cache_(b, j) / clipped_pred;
                    }
                }
            }
            
            if (reduction_ == "mean")
            {
                #pragma omp parallel for collapse(2) schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    for (int j = 0; j < num_classes; ++j)
                    {
                        gradients(b, j) /= batch_size;
                    }
                }
            }
            
            return gradients;
        }
        Eigen::Tensor<float, 3> CategoricalCrossEntropy::backward(const Eigen::Tensor<float, 3>& predictions, const Eigen::Tensor<int, 2>& targets) 
        {
            int batch_size = predictions.dimension(0);
            int seq_len = predictions.dimension(1);
            int num_classes = predictions.dimension(2);

            if (targets.dimension(0) != batch_size || targets.dimension(1) != seq_len)
                throw std::runtime_error("CategoricalCrossEntropy: Shape mismatch in backward");

            int total = batch_size * seq_len;
            Eigen::Tensor<float, 2> pred_2d(total, num_classes);
            Eigen::Tensor<float, 2> one_hot(total, num_classes);
            one_hot.setZero();

            for (int b = 0; b < batch_size; ++b)
            {
                for (int t = 0; t < seq_len; ++t)
                {
                    int idx = b * seq_len + t;
                    int cls = targets(b, t);
                    if (cls < 0 || cls >= num_classes)
                        throw std::runtime_error("CategoricalCrossEntropy: Target index out of range");
                    one_hot(idx, cls) = 1.0f;
                    for (int c = 0; c < num_classes; ++c)
                        pred_2d(idx, c) = predictions(b, t, c);
                }
            }

            forward(pred_2d, one_hot);
            Eigen::Tensor<float, 2> grad_2d = backward(pred_2d, one_hot);

            Eigen::Tensor<float, 3> grad(batch_size, seq_len, num_classes);
            for (int b = 0; b < batch_size; ++b)
                for (int t = 0; t < seq_len; ++t)
                {
                    int idx = b * seq_len + t;
                    for (int c = 0; c < num_classes; ++c)
                        grad(b, t, c) = grad_2d(idx, c);
                }

            return grad;
        }
    }
}