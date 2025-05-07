#include <Eigen/Dense>
#include "activations.hpp"




namespace CppNet
{
    Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& z)
    {
        // validate input
        if (z.size() == 0) 
        {
            throw std::runtime_error("Empty input matrix!");
        }

        in_cache_ = z;
        return z.array().cwiseMax(0.0);
    }

    Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& da)
    {
        // validate input
        if (da.rows() != in_cache_.rows() || da.cols() != in_cache_.cols() || da.size() == 0) 
        {
            throw std::runtime_error("Shape mismatch or empty input: da and in_cache_ must have equal non-zero size!");
        }

        // compute gradient: da * (z > 0)
        Eigen::MatrixXd grad = da;
        grad.array() *= (in_cache_.array() > 0).cast<double>();
        return grad;
    }

    Eigen::MatrixXd Sigmoid::forward(const Eigen::MatrixXd& z)
    {
        // validate input
        if (z.size() == 0) 
        {
            throw std::runtime_error("Empty input matrix!");
        }

        in_cache_ = z;
        return 1.0 / (1.0 + (-z.array()).exp());
    }

    Eigen::MatrixXd Sigmoid::backward(const Eigen::MatrixXd& da)
    {
        // validate input
        if (da.rows() != in_cache_.rows() || da.cols() != in_cache_.cols() || da.size() == 0) 
        {
            throw std::runtime_error("Shape mismatch or empty input: da and in_cache_ must have equal non-zero size!");
        }

        // compute gradient: da * sigmoid(z) * (1 - sigmoid(z))
        Eigen::MatrixXd sig = 1.0 / (1.0 + (-in_cache_.array()).exp());
        return da.array() * sig.array() * (1.0 - sig.array());
    }
}