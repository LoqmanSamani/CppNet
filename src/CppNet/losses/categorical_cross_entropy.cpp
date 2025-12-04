#include <cmath>
#include <Eigen/Dense>
#include "CppNet/losses/categorical_cross_entropy.hpp"





namespace CppNet
{
    namespace Losses
    {
        /************************************** CategoricalCrossEntropy Loss Function *************************************/
        CategoricalCrossEntropy::CategoricalCrossEntropy(const std::string& reduction, bool from_logits, double label_smoothing) 
            : reduction_(reduction), from_logits_(from_logits), label_smoothing_(label_smoothing) {}

        // helper method to set number of threads
        void CategoricalCrossEntropy::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }

        double CategoricalCrossEntropy::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets) 
        {
            // TODO: implement CrossEntropy for class indices: -sum(log(softmax(pred)[target]))
            return 0.0;
        }



        double CategoricalCrossEntropy::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets)
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
                double smoothing_factor = label_smoothing_ / num_classes;
                targets_cache_ = targets * (1.0 - label_smoothing_) + smoothing_factor;
            }
            
            // handle predictions based on from_logits flag
            if (from_logits_)
            {
                // apply softmax and cache the result
                softmax_cache_ = Eigen::Tensor<double, 2>(batch_size, num_classes);
                
                #pragma omp parallel for schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    // find max for numerical stability
                    double max_val = predictions(b, 0);
                    for (int j = 1; j < num_classes; ++j)
                    {
                        if (predictions(b, j) > max_val)
                            max_val = predictions(b, j);
                    }
                    
                    // compute exp(x - max) and sum
                    double sum_exp = 0.0;
                    for (int j = 0; j < num_classes; ++j)
                    {
                        double exp_val = std::exp(predictions(b, j) - max_val);
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
            double total_loss = 0.0;
            const double epsilon = 1e-15; // small value to prevent log(0)
            
            #pragma omp parallel for reduction(+:total_loss) schedule(static)
            for (int b = 0; b < batch_size; ++b)
            {
                double sample_loss = 0.0;
                for (int j = 0; j < num_classes; ++j)
                {
                    // clip predictions to prevent log(0)
                    double clipped_pred = std::max(epsilon, std::min(1.0 - epsilon, softmax_cache_(b, j)));
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

        double CategoricalCrossEntropy::forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<int, 2>& targets) 
        {
            return 0.0;
        }

        Eigen::Tensor<double, 2> CategoricalCrossEntropy::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets) 
        {
            // TODO: implement CrossEntropy gradient: softmax(pred) - one_hot(target)
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 2> CategoricalCrossEntropy::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets)
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
            
            Eigen::Tensor<double, 2> gradients(batch_size, num_classes);
            
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
                const double epsilon = 1e-15;
                
                #pragma omp parallel for collapse(2) schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    for (int j = 0; j < num_classes; ++j)
                    {
                        // clip predictions to prevent division by 0
                        double clipped_pred = std::max(epsilon, std::min(1.0 - epsilon, softmax_cache_(b, j)));
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
        Eigen::Tensor<double, 3> CategoricalCrossEntropy::backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<int, 2>& targets) 
        {
            Eigen::Tensor<double, 3> grad(predictions.dimension(0), predictions.dimension(1), predictions.dimension(2));
            grad.setZero();
            return grad;
        }
    }
}