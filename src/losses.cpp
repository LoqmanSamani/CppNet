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

        /************************************** CrossEntropy *************************************/
        CrossEntropy::CrossEntropy(const std::string& reduction, bool from_logits, double label_smoothing) 
            : reduction(reduction), from_logits(from_logits), label_smoothing(label_smoothing) 
        {
            // Store parameters
        }

        double CrossEntropy::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets) 
        {
            // TODO: Implement CrossEntropy for class indices: -sum(log(softmax(pred)[target]))
            return 0.0;
        }

        double CrossEntropy::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            // TODO: Implement CrossEntropy for one-hot targets: -sum(target * log(softmax(pred)))
            return 0.0;
        }

        double CrossEntropy::forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<int, 2>& targets) 
        {
            return 0.0;
        }

        Eigen::Tensor<double, 2> CrossEntropy::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets) 
        {
            // TODO: Implement CrossEntropy gradient: softmax(pred) - one_hot(target)
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 2> CrossEntropy::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 3> CrossEntropy::backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<int, 2>& targets) 
        {
            Eigen::Tensor<double, 3> grad(predictions.dimension(0), predictions.dimension(1), predictions.dimension(2));
            grad.setZero();
            return grad;
        }

        /************************************** BinaryCrossEntropy *************************************/
        BinaryCrossEntropy::BinaryCrossEntropy(const std::string& reduction, bool from_logits, double pos_weight) 
            : reduction(reduction), from_logits(from_logits), pos_weight(pos_weight) 
        {
            // Store parameters
        }

        double BinaryCrossEntropy::forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement BCE: -sum(target * log(sigmoid(pred)) + (1-target) * log(1-sigmoid(pred)))
            return 0.0;
        }

        double BinaryCrossEntropy::forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            return 0.0;
        }

        Eigen::Tensor<double, 1> BinaryCrossEntropy::backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) 
        {
            // TODO: Implement BCE gradient: sigmoid(pred) - target
            Eigen::Tensor<double, 1> grad(predictions.dimension(0));
            grad.setZero();
            return grad;
        }

        Eigen::Tensor<double, 2> BinaryCrossEntropy::backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) 
        {
            Eigen::Tensor<double, 2> grad(predictions.dimension(0), predictions.dimension(1));
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