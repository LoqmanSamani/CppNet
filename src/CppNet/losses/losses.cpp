#include <cmath>
#include <Eigen/Dense>
#include "losses.hpp"

namespace CppNet
{
    namespace Losses
    {
        /******************************BinaryCrossEntropy implementation**********************************/

        BinaryCrossEntropy::BinaryCrossEntropy(const std::string& reduction, bool from_logits, double pos_weight) 
            : reduction_(reduction), from_logits_(from_logits), pos_weight_(pos_weight) 
        {
            if (reduction_ != "mean" && reduction_ != "sum" && reduction_ != "none")
            {
                throw std::runtime_error("Invalid reduction method: " + reduction_ + ". Must be 'mean', 'sum', or 'none'.");
            }
        }
        
        void BinaryCrossEntropy::validate_inputs(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets)
        {
            // check if tensors have the same dimensions
            if (predictions.dimension(0) != targets.dimension(0) || predictions.dimension(1) != targets.dimension(1))
            {
                throw std::runtime_error("Shape mismatch: predictions and targets must have the same dimensions!");
            }
            // check if tensors are not empty
            if (targets.size() == 0)
            {
                throw std::runtime_error("Empty input: predictions and targets cannot be empty!");
            }
            // validate binary labels (0 or 1) - parallelized validation
            bool valid_labels = true;
            const int rows = targets.dimension(0);
            const int cols = targets.dimension(1);

            #pragma omp parallel for collapse(2) reduction(&&:valid_labels)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    double val = targets(i, j);
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

        // helper method to set number of threads
        void BinaryCrossEntropy::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }

        double BinaryCrossEntropy::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets)
        {
            validate_inputs(predictions, targets);
            
            // get dimensions
            const int rows = predictions.dimension(0);
            const int cols = predictions.dimension(1);
            const int total_size = rows * cols;

            double total_loss = 0.0;
            
            #pragma omp parallel for collapse(2) reduction(+:total_loss)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    // clip values to [1e-15, 1 - 1e-15] to avoid log(0) and division by 0
                    double pred_clipped = std::max(1e-15, std::min(predictions(i, j), 1.0 - 1e-15));
                    double target_val = targets(i, j);
                    
                    // binary cross-entropy: -[y * log(y_hat) + (1-y) * log(1-y_hat)]
                    double loss_val = -(target_val * std::log(pred_clipped) + 
                                       (1.0 - target_val) * std::log(1.0 - pred_clipped));
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
                // For "none", we return the mean as a placeholder since the return type is double
                return total_loss / total_size;
            }
        }

        Eigen::Tensor<double, 2> BinaryCrossEntropy::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets)
        {
            validate_inputs(predictions, targets);

            // get dimensions
            const int rows = predictions.dimension(0);
            const int cols = predictions.dimension(1);
            const int total_size = rows * cols;
            
            // Create output gradient tensor
            Eigen::Tensor<double, 2> grad(rows, cols);
            
            // Parallelize gradient computation
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    // clip values to [1e-15, 1 - 1e-15] to avoid log(0) and division by 0
                    double pred_clipped = std::max(1e-15, std::min(predictions(i, j), 1.0 - 1e-15));
                    double target_val = targets(i, j);
                    
                    // gradient: (y_hat - y) / (y_hat * (1 - y_hat))
                    double grad_val = (pred_clipped - target_val) / (pred_clipped * (1.0 - pred_clipped));
                    
                    if (reduction_ == "mean")
                    {
                        grad_val /= total_size;
                    }
                    // For "sum" and "none", no additional scaling needed
                    
                    grad(i, j) = grad_val;
                }
            }
            
            return grad;
        }
        

        /************************************** MSE (Mean Squared Error) *************************************/
        MSE::MSE(const std::string& reduction) : reduction(reduction) 
        {
            // Store reduction parameter
        }

        double MSE::forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement MSE: (1/n) * sum((pred - target)²)
            return 0.0;
        }

        double MSE::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            return 0.0;
        }

        double MSE::forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets) 
        {
            return 0.0;
        }

        Eigen::Tensor<double, 1> MSE::backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement MSE gradient: 2 * (pred - target) / n
            Eigen::Tensor<double, 1> grad(predictions.dimension(0));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 2> MSE::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 3> MSE::backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets) 
        {
            Eigen::Tensor<double, 3> grad(predictions.dimension(0), predictions.dimension(1), predictions.dimension(2));
            grad.setZero();
            return grad;
        }

        /************************************** MAE (Mean Absolute Error) *************************************/
        MAE::MAE(const std::string& reduction) : reduction(reduction) 
        {
            // Store reduction parameter
        }

        double MAE::forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement MAE: (1/n) * sum(|pred - target|)
            return 0.0;
        }

        double MAE::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            return 0.0;
        }

        double MAE::forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets) 
        {
            return 0.0;
        }

        Eigen::Tensor<double, 1> MAE::backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement MAE gradient: sign(pred - target) / n
            Eigen::Tensor<double, 1> grad(predictions.dimension(0));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 2> MAE::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 3> MAE::backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets) 
        {
            Eigen::Tensor<double, 3> grad(predictions.dimension(0), predictions.dimension(1), predictions.dimension(2));
            grad.setZero();
            return grad;
        }

        /************************************** CategoricalCrossEntropy *************************************/
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
            // TODO: Implement CrossEntropy for class indices: -sum(log(softmax(pred)[target]))
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
            
            // Process targets with label smoothing and cache
            targets_cache_ = targets;
            if (label_smoothing_ > 0.0)
            {
                double smoothing_factor = label_smoothing_ / num_classes;
                targets_cache_ = targets * (1.0 - label_smoothing_) + smoothing_factor;
            }
            
            // Handle predictions based on from_logits flag
            if (from_logits_)
            {
                // Apply softmax and cache the result
                softmax_cache_ = Eigen::Tensor<double, 2>(batch_size, num_classes);
                
                #pragma omp parallel for schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    // Find max for numerical stability
                    double max_val = predictions(b, 0);
                    for (int j = 1; j < num_classes; ++j)
                    {
                        if (predictions(b, j) > max_val)
                            max_val = predictions(b, j);
                    }
                    
                    // Compute exp(x - max) and sum
                    double sum_exp = 0.0;
                    for (int j = 0; j < num_classes; ++j)
                    {
                        double exp_val = std::exp(predictions(b, j) - max_val);
                        softmax_cache_(b, j) = exp_val;
                        sum_exp += exp_val;
                    }
                    
                    // Normalize to get probabilities
                    for (int j = 0; j < num_classes; ++j)
                    {
                        softmax_cache_(b, j) /= sum_exp;
                    }
                }
            }
            else
            {
                // Input is already probabilities, just cache them
                softmax_cache_ = predictions;
            }
            
            // Compute cross-entropy loss using cached softmax: -sum(target * log(pred))
            double total_loss = 0.0;
            const double epsilon = 1e-15; // Small value to prevent log(0)
            
            #pragma omp parallel for reduction(+:total_loss) schedule(static)
            for (int b = 0; b < batch_size; ++b)
            {
                double sample_loss = 0.0;
                for (int j = 0; j < num_classes; ++j)
                {
                    // Clip predictions to prevent log(0)
                    double clipped_pred = std::max(epsilon, std::min(1.0 - epsilon, softmax_cache_(b, j)));
                    sample_loss -= targets_cache_(b, j) * std::log(clipped_pred);
                }
                total_loss += sample_loss;
            }
            
            // Apply reduction
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
            // TODO: Implement CrossEntropy gradient: softmax(pred) - one_hot(target)
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 2> CategoricalCrossEntropy::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets)
        {
            // Check if forward was called first (caches should be populated)
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
                // Use cached softmax: gradient = softmax(logits) - targets
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
                // Input was probabilities: gradient = -targets / predictions
                const double epsilon = 1e-15;
                
                #pragma omp parallel for collapse(2) schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    for (int j = 0; j < num_classes; ++j)
                    {
                        // Clip predictions to prevent division by 0
                        double clipped_pred = std::max(epsilon, std::min(1.0 - epsilon, softmax_cache_(b, j)));
                        gradients(b, j) = -targets_cache_(b, j) / clipped_pred;
                    }
                }
            }
            
            // Apply reduction scaling
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

        /************************************** HuberLoss *************************************/
        HuberLoss::HuberLoss(double delta, const std::string& reduction) : delta(delta), reduction(reduction) 
        {
            // Store parameters
        }

        double HuberLoss::forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement Huber: 0.5*(x²) if |x| < delta, else delta*(|x| - 0.5*delta) where x = pred-target
            return 0.0;
        }

        double HuberLoss::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            return 0.0;
        }

        double HuberLoss::forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets) 
        {
            return 0.0;
        }

        Eigen::Tensor<double, 1> HuberLoss::backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement Huber gradient: x if |x| < delta, else delta*sign(x)
            Eigen::Tensor<double, 1> grad(predictions.dimension(0));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 2> HuberLoss::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 3> HuberLoss::backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets) 
        {
            Eigen::Tensor<double, 3> grad(predictions.dimension(0), predictions.dimension(1), predictions.dimension(2));
            grad.setZero();
            return grad;
        }

        /************************************** KLDivergence *************************************/
        KLDivergence::KLDivergence(const std::string& reduction, bool log_target) : reduction(reduction), log_target(log_target) 
        {
            // Store parameters
        }

        double KLDivergence::forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement KL Divergence: sum(target * log(target / pred))
            return 0.0;
        }

        double KLDivergence::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            return 0.0;
        }

        double KLDivergence::forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets) 
        {
            return 0.0;
        }

        Eigen::Tensor<double, 1> KLDivergence::backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement KL gradient: -target / pred
            Eigen::Tensor<double, 1> grad(predictions.dimension(0));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 2> KLDivergence::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 3> KLDivergence::backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets) 
        {
            Eigen::Tensor<double, 3> grad(predictions.dimension(0), predictions.dimension(1), predictions.dimension(2));
            grad.setZero();
            return grad;
        }

        /************************************** FocalLoss *************************************/
        FocalLoss::FocalLoss(double alpha, double gamma, const std::string& reduction) 
            : alpha(alpha), gamma(gamma), reduction(reduction) 
        {
            // Store parameters
        }

        double FocalLoss::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets) 
        {
            // TODO: Implement Focal Loss: -alpha * (1-pt)^gamma * log(pt)
            return 0.0;
        }

        double FocalLoss::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            return 0.0;
        }

        Eigen::Tensor<double, 2> FocalLoss::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets) 
        {
            // TODO: Implement Focal Loss gradient
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 2> FocalLoss::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }
    }
}