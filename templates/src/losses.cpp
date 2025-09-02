#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "losses.hpp"
#include <cmath>




namespace CppNet
{
    namespace Losses
    {
        // BinaryCrossEntropy implementation
        void BinaryCrossEntropy::validate_inputs(const Eigen::Tensor<double, 2>& y, const Eigen::Tensor<double, 2>& y_hat)
        {
            // check if tensors have the same dimensions
            if (y.dimension(0) != y_hat.dimension(0) || y.dimension(1) != y_hat.dimension(1))
            {
            throw std::runtime_error("Shape mismatch: y and y_hat must have the same dimensions!");
            }
            // check if tensors are not empty
            if (y.size() == 0)
            {
            throw std::runtime_error("Empty input: y and y_hat cannot be empty!");
            }
            // validate binary labels (0 or 1) - using tensor operations
            bool valid_labels = true;
            for (int i = 0; i < y.dimension(0) && valid_labels; ++i)
            {
            for (int j = 0; j < y.dimension(1) && valid_labels; ++j)
            {
            double val = y(i, j);
            if (val != 0.0 && val != 1.0)
            {
            valid_labels = false;
            }
            }
            }
            if (!valid_labels)
            {
            throw std::runtime_error("y must contain binary labels (0 or 1)!");
            }
        }

        double BinaryCrossEntropy::forward(const Eigen::Tensor<double, 2>& y, const Eigen::Tensor<double, 2>& y_hat)
        {
            validate_inputs(y, y_hat);
            // create matrix views of tensor data (zero-copy)
            Eigen::Map<const Eigen::MatrixXd> y_map(y.data(), y.dimension(0), y.dimension(1));
            Eigen::Map<const Eigen::MatrixXd> y_hat_map(y_hat.data(), y_hat.dimension(0), y_hat.dimension(1));
            // clip values to [1e-15, 1 - 1e-15] to avoid log(0) and division by 0
            Eigen::MatrixXd y_hat_clipped = y_hat_map.array().cwiseMax(1e-15).cwiseMin(1.0 - 1e-15);
            // binary cross-entropy: -[y * log(y_hat) + (1-y) * log(1-y_hat)]
            Eigen::MatrixXd loss_matrix = -(y_map.array() * y_hat_clipped.array().log() + (1.0 - y_map.array()) * (1.0 - y_hat_clipped.array()).log());
            return loss_matrix.sum() / y.size();
        }

        Eigen::Tensor<double, 2> BinaryCrossEntropy::backward(const Eigen::Tensor<double, 2>& y_hat, const Eigen::Tensor<double, 2>& y)
        {
            validate_inputs(y, y_hat);
            // create matrix views
            Eigen::Map<const Eigen::MatrixXd> y_map(y.data(), y.dimension(0), y.dimension(1));
            Eigen::Map<const Eigen::MatrixXd> y_hat_map(y_hat.data(), y_hat.dimension(0), y_hat.dimension(1));
            // clip values to [1e-15, 1 - 1e-15] to avoid log(0) and division by 0
            Eigen::MatrixXd y_hat_clipped = y_hat_map.array().cwiseMax(1e-15).cwiseMin(1.0 - 1e-15);
            // compute gradient: grad_matrix = (y_hat - y) / (y_hat * (1 - y_hat)) / N
            Eigen::MatrixXd grad_matrix = (y_hat_clipped - y_map).array() / (y_hat_clipped.array() * (1.0 - y_hat_clipped.array())) / y.size();
            // convert back to tensor
            Eigen::Tensor<double, 2> grad_tensor(y.dimension(0), y.dimension(1));
            Eigen::Map<Eigen::MatrixXd> grad_tensor_map(grad_tensor.data(), y.dimension(0), y.dimension(1));
            grad_tensor_map = grad_matrix;
            return grad_tensor;
        }

        // CategoricalCrossEntropy implementation
        void CategoricalCrossEntropy::validate_inputs(const Eigen::Tensor<double, 2>& y, const Eigen::Tensor<double, 2>& y_hat)
        {
            // check if tensors have the same dimensions
            if (y.dimension(0) != y_hat.dimension(0) || y.dimension(1) != y_hat.dimension(1))
            {
            throw std::runtime_error("Shape mismatch: y and y_hat must have the same dimensions!");
            }
            // check if tensors are not empty
            if (y.size() == 0)
            {
            throw std::runtime_error("Empty input: y and y_hat cannot be empty!");
            }
            
            // validate one-hot encoding for y (each row should sum to 1 and contain only 0s and 1s)
            for (int i = 0; i < y.dimension(0); ++i)
            {
            double row_sum = 0.0;
            bool valid_one_hot = true;
            for (int j = 0; j < y.dimension(1); ++j)
            {
            double val = y(i, j);
            if (val != 0.0 && val != 1.0)
            {
            valid_one_hot = false;
            break;
            }
            row_sum += val;
            }
            /*
            if (!valid_one_hot || std::abs(row_sum - 1.0) > 1e-9)
            {
            throw std::runtime_error("y must be one-hot encoded (each row should contain exactly one 1 and the rest 0s)!");
            }
            */
            }
            
            // validate that y_hat contains probabilities (should be >= 0 and each row should sum to ~1)
            for (int i = 0; i < y_hat.dimension(0); ++i)
            {
            double row_sum = 0.0;
            for (int j = 0; j < y_hat.dimension(1); ++j)
            {
            double val = y_hat(i, j);
            if (val < 0.0)
            {
            throw std::runtime_error("y_hat must contain non-negative values (probabilities)!");
            }
            row_sum += val;
            }
            if (std::abs(row_sum - 1.0) > 1e-6)
            {
            throw std::runtime_error("y_hat rows must sum to 1 (softmax outputs)!");
            }
            }
        }

        double CategoricalCrossEntropy::forward(const Eigen::Tensor<double, 2>& y, const Eigen::Tensor<double, 2>& y_hat)
        {
            validate_inputs(y, y_hat);
            
            // create matrix views of tensor data (zero-copy)
            Eigen::Map<const Eigen::MatrixXd> y_map(y.data(), y.dimension(0), y.dimension(1));
            Eigen::Map<const Eigen::MatrixXd> y_hat_map(y_hat.data(), y_hat.dimension(0), y_hat.dimension(1));
            
            // clip predictions to avoid log(0)
            Eigen::MatrixXd y_hat_clipped = y_hat_map.array().cwiseMax(1e-15);
            
            // categorical cross-entropy: -sum(y * log(y_hat))
            // Since y is one-hot, this reduces to -log(y_hat[correct_class]) for each sample
            Eigen::MatrixXd loss_matrix = -(y_map.array() * y_hat_clipped.array().log());
            
            // sum over all samples and classes, then average over batch size
            return loss_matrix.sum() / y.dimension(0);
        }

        Eigen::Tensor<double, 2> CategoricalCrossEntropy::backward(const Eigen::Tensor<double, 2>& y_hat, const Eigen::Tensor<double, 2>& y)
        {
            validate_inputs(y, y_hat);
            
            // create matrix views
            Eigen::Map<const Eigen::MatrixXd> y_map(y.data(), y.dimension(0), y.dimension(1));
            Eigen::Map<const Eigen::MatrixXd> y_hat_map(y_hat.data(), y_hat.dimension(0), y_hat.dimension(1));
            
            // clip predictions to avoid division by 0
            Eigen::MatrixXd y_hat_clipped = y_hat_map.array().cwiseMax(1e-15);
            
            // gradient of categorical cross-entropy: -y / y_hat / batch_size
            // For softmax + categorical cross-entropy, this simplifies to: (y_hat - y) / batch_size
            // But we'll use the general form for numerical stability
            Eigen::MatrixXd grad_matrix = -(y_map.array() / y_hat_clipped.array()) / y.dimension(0);
            
            // convert back to tensor
            Eigen::Tensor<double, 2> grad_tensor(y.dimension(0), y.dimension(1));
            Eigen::Map<Eigen::MatrixXd> grad_tensor_map(grad_tensor.data(), y.dimension(0), y.dimension(1));
            grad_tensor_map = grad_matrix;
            
            return grad_tensor;
        }
    }
}

