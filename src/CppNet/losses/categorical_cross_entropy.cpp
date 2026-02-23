#include <cmath>
#include <Eigen/Dense>
#include "CppNet/losses/categorical_cross_entropy.hpp"





namespace CppNet
{
    namespace Losses
    {
        CategoricalCrossEntropy::CategoricalCrossEntropy(const std::string& reduction, bool from_logits, float label_smoothing) 
            : reduction_(reduction), from_logits_(from_logits), label_smoothing_(label_smoothing) {}

        // helper method to set number of threads
        void CategoricalCrossEntropy::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
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

            // Convert class indices to one-hot and delegate
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
            
            // process targets with label smoothing and cache
            targets_cache_ = targets;
            if (label_smoothing_ > 0.0)
            {
                float smoothing_factor = label_smoothing_ / num_classes;
                targets_cache_ = targets * (1.0f - label_smoothing_) + smoothing_factor;
            }
            
            // handle predictions based on from_logits flag
            if (from_logits_)
            {
                // apply softmax and cache the result
                softmax_cache_ = Eigen::Tensor<float, 2>(batch_size, num_classes);
                
                #pragma omp parallel for schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    // find max for numerical stability
                    float max_val = predictions(b, 0);
                    for (int j = 1; j < num_classes; ++j)
                    {
                        if (predictions(b, j) > max_val)
                            max_val = predictions(b, j);
                    }
                    
                    // compute exp(x - max) and sum
                    float sum_exp = 0.0f;
                    for (int j = 0; j < num_classes; ++j)
                    {
                        float exp_val = std::exp(predictions(b, j) - max_val);
                        softmax_cache_(b, j) = exp_val;
                        sum_exp += exp_val;
                    }
                    
                    // normalize to get probabilities
                    for (int j = 0; j < num_classes; ++j)
                    {
                        softmax_cache_(b, j) /= sum_exp;
                    }
                }
            }
            else
            {
                // input is already probabilities, just cache them
                softmax_cache_ = predictions;
            }
            
            // compute cross-entropy loss using cached softmax: -sum(target * log(pred))
            float total_loss = 0.0f;
            const float epsilon = 1e-15f; // small value to prevent log(0)
            
            #pragma omp parallel for reduction(+:total_loss) schedule(static)
            for (int b = 0; b < batch_size; ++b)
            {
                float sample_loss = 0.0f;
                for (int j = 0; j < num_classes; ++j)
                {
                    // clip predictions to prevent log(0)
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
            // predictions: [batch, seq_len, num_classes], targets: [batch, seq_len] (class indices)
            if (predictions.size() == 0 || targets.size() == 0)
                throw std::runtime_error("CategoricalCrossEntropy: Empty input tensors");

            int batch_size = predictions.dimension(0);
            int seq_len = predictions.dimension(1);
            int num_classes = predictions.dimension(2);

            if (targets.dimension(0) != batch_size || targets.dimension(1) != seq_len)
                throw std::runtime_error("CategoricalCrossEntropy: Shape mismatch between predictions and targets");

            // Reshape to 2D [batch*seq_len, num_classes] and [batch*seq_len]
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

            // Convert class indices to one-hot and delegate
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
            // check if forward was called first (caches should be populated)
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
            
            if (from_logits_)
            {
                // use cached softmax: gradient = softmax(logits) - targets
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
                // input was probabilities: gradient = -targets / predictions
                const float epsilon = 1e-15f;
                
                #pragma omp parallel for collapse(2) schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    for (int j = 0; j < num_classes; ++j)
                    {
                        // clip predictions to prevent division by 0
                        float clipped_pred = std::max(epsilon, std::min(1.0f - epsilon, softmax_cache_(b, j)));
                        gradients(b, j) = -targets_cache_(b, j) / clipped_pred;
                    }
                }
            }
            
            // apply reduction scaling
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

            // Reshape to 2D, compute gradient, reshape back
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

            // Need to call forward on 2D to populate caches, then backward
            forward(pred_2d, one_hot);
            Eigen::Tensor<float, 2> grad_2d = backward(pred_2d, one_hot);

            // Reshape back to 3D
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