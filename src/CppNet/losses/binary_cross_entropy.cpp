#include <cmath>
#include <Eigen/Dense>
#include "CppNet/losses/binary_cross_entropy.hpp"




namespace CppNet
{
    namespace Losses
    {
        /******************************BinaryCrossEntropy Loss Function********************************/

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
            
            // create output gradient tensor
            Eigen::Tensor<double, 2> grad(rows, cols);
            
            // parallelize gradient computation
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
                    // for "sum" and "none", no additional scaling needed
                    
                    grad(i, j) = grad_val;
                }
            }
            
            return grad;
        }
    }
}