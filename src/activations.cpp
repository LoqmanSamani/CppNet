#include <Eigen/Dense>
#include "activations.hpp"




namespace CppNet
{
    Eigen::Tensor<double, 2> ReLU::forward(const Eigen::Tensor<double, 2>& z)
    {
        // validate input
        if (z.size() == 0) 
        {
            throw std::runtime_error("Empty input matrix!");
        }

        in_cache_ = z;
        return z.cwiseMax(z.constant(0.0));
    }

    Eigen::Tensor<double, 2> ReLU::backward(const Eigen::Tensor<double, 2>& da)
    {
        // validate input
        if (da.dimension(0) != in_cache_.dimension(0) || da.dimension(1) != in_cache_.dimension(1) || da.size() == 0) 
        {
            throw std::runtime_error("Shape mismatch or empty input: da and in_cache_ must have equal non-zero size!");
        }

        // compute gradient: da * (z > 0)
        Eigen::Tensor<double, 2> grad = da;
        grad = da * (in_cache_ > in_cache_.constant(0.0)).template cast<double>();

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