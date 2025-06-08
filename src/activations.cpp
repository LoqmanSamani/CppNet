#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "activations.hpp"



namespace CppNet
{
    namespace Activations
    {
        Eigen::Tensor<double, 2> ReLU::forward(const Eigen::Tensor<double, 2>& z)
        {
            // validate input
            if (z.size() == 0)
            {
                throw std::runtime_error("Empty input tensor!");
            }
            
            in_cache_ = z;
            return z.cwiseMax(z.constant(0.0));
        }

        Eigen::Tensor<double, 2> ReLU::backward(const Eigen::Tensor<double, 2>& da)
        {
            // validate input
            if (da.dimension(0) != in_cache_.dimension(0) || 
                da.dimension(1) != in_cache_.dimension(1) || 
                da.size() == 0)
            {
                throw std::runtime_error("Shape mismatch or empty input: da and in_cache_ must have equal non-zero size!");
            }
            
            // compute gradient: da * (z > 0)
            // create a tensor of ones and zeros based on the condition
            Eigen::Tensor<double, 2> mask = (in_cache_ > in_cache_.constant(0.0)).template cast<double>();
            return da * mask;
        }

        Eigen::Tensor<double, 2> Sigmoid::forward(const Eigen::Tensor<double, 2>& z)
        {
            // validate input
            if (z.size() == 0)
            {
                throw std::runtime_error("Empty input tensor!");
            }
            
            in_cache_ = z;
            
            // compute sigmoid: 1 / (1 + exp(-z))
            // for tensors, we need to use unaryExpr for element-wise operations
            sigmoid_cache_ = z.unaryExpr([](double x) { 
                return 1.0 / (1.0 + std::exp(-x)); 
            });
            
            return sigmoid_cache_;
        }

        Eigen::Tensor<double, 2> Sigmoid::backward(const Eigen::Tensor<double, 2>& da)
        {
            // validate input
            if (da.dimension(0) != in_cache_.dimension(0) || 
                da.dimension(1) != in_cache_.dimension(1) || 
                da.size() == 0)
            {
                throw std::runtime_error("Shape mismatch or empty input: da and in_cache_ must have equal non-zero size!");
            }
            
            // compute gradient: da * sigmoid(z) * (1 - sigmoid(z))
            // Use cached sigmoid output for efficiency
            Eigen::Tensor<double, 2> one_minus_sigmoid = sigmoid_cache_.unaryExpr([](double x) { 
                return 1.0 - x; 
            });
            
            return da * sigmoid_cache_ * one_minus_sigmoid;
        }
    }
}