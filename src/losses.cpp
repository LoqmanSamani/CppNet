#include <Eigen/Dense>
#include "losses.hpp"




namespace CppNet
{
    double BinaryCrossEntropy::forward(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_hat)
    {
        // validate inputs
        if (y.size() != y_hat.size() || y.size() == 0) 
        {
            throw std::runtime_error("Shape mismatch or empty input: y and y_hat must have equal non-zero size!");
        }

        if (!y.array().isApprox(y.array().round())) 
        {
            throw std::runtime_error("y must contain binary labels (0 or 1)!");
        }

        // clip y_hat to avoid log(0) or log(1 - 1)
        Eigen::MatrixXd y_hat_clipped = y_hat.array().cwiseMax(1e-15).cwiseMin(1.0 - 1e-15);

        // compute binary cross-entropy loss: -1/N * sum(y * log(y_hat) + (1-y) * log(1-y_hat))
        Eigen::MatrixXd l = -(y.array() * y_hat_clipped.array().log() + 
                              (1.0 - y.array()) * (1.0 - y_hat_clipped.array()).log());
        
        return l.sum() / y.size();
    }

    Eigen::MatrixXd BinaryCrossEntropy::backward(const Eigen::MatrixXd& y_hat, const Eigen::MatrixXd& y)
    {
        // validate inputs
        if (y.size() != y_hat.size() || y.size() == 0) 
        {
            throw std::runtime_error("Shape mismatch or empty input: y and y_hat must have equal non-zero size!");
        }

        if (!y.array().isApprox(y.array().round())) 
        {
            throw std::runtime_error("y must contain binary labels (0 or 1)!");
        }

        // clip y_hat to avoid division by zero
        Eigen::MatrixXd y_hat_clipped = y_hat.array().cwiseMax(1e-15).cwiseMin(1.0 - 1e-15);

        // compute gradient: 1/N * (y_hat - y) / (y_hat * (1 - y_hat))
        return (y_hat_clipped - y).array() / (y_hat_clipped.array() * (1.0 - y_hat_clipped.array())) / y.size();
    }
}