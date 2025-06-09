#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "activations.hpp"
#include <cmath>

namespace CppNet
{
    namespace Activations
    {
        // 2D tensor version 
        Eigen::Tensor<double, 2> ReLU::forward(const Eigen::Tensor<double, 2>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("Empty input tensor!");
            }
            in_cache_2d_ = z;
            return z.cwiseMax(z.constant(0.0));
        }

        Eigen::Tensor<double, 2> ReLU::backward(const Eigen::Tensor<double, 2>& da)
        {
            if (da.dimension(0) != in_cache_2d_.dimension(0) ||
                da.dimension(1) != in_cache_2d_.dimension(1) ||
                da.size() == 0)
            {
                throw std::runtime_error("Shape mismatch or empty input: da and in_cache_ must have equal non-zero size!");
            }
            
            Eigen::Tensor<double, 2> mask = (in_cache_2d_ > in_cache_2d_.constant(0.0)).template cast<double>();
            return da * mask;
        }

        // scalar version
        double ReLU::forward(double z)
        {
            return std::max(0.0, z);
        }

        // 4D tensor version
        Eigen::Tensor<double, 4> ReLU::forward(const Eigen::Tensor<double, 4>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("Empty input tensor!");
            }
            in_cache_4d_ = z;
            return z.cwiseMax(z.constant(0.0));
        }

        Eigen::Tensor<double, 4> ReLU::backward(const Eigen::Tensor<double, 4>& da)
        {
            if (da.dimension(0) != in_cache_4d_.dimension(0) ||
                da.dimension(1) != in_cache_4d_.dimension(1) ||
                da.dimension(2) != in_cache_4d_.dimension(2) ||
                da.dimension(3) != in_cache_4d_.dimension(3) ||
                da.size() == 0)
            {
                throw std::runtime_error("Shape mismatch or empty input: da and in_cache_ must have equal non-zero size!");
            }
            
            Eigen::Tensor<double, 4> mask = (in_cache_4d_ > in_cache_4d_.constant(0.0)).template cast<double>();
            return da * mask;
        }

        // 2D tensor version
        Eigen::Tensor<double, 2> Sigmoid::forward(const Eigen::Tensor<double, 2>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("Empty input tensor!");
            }
            in_cache_2d_ = z;
            
            sigmoid_cache_2d_ = z.unaryExpr([](double x) {
                return 1.0 / (1.0 + std::exp(-x));
            });
            return sigmoid_cache_2d_;
        }

        Eigen::Tensor<double, 2> Sigmoid::backward(const Eigen::Tensor<double, 2>& da)
        {
            if (da.dimension(0) != in_cache_2d_.dimension(0) ||
                da.dimension(1) != in_cache_2d_.dimension(1) ||
                da.size() == 0)
            {
                throw std::runtime_error("Shape mismatch or empty input: da and in_cache_ must have equal non-zero size!");
            }
            
            Eigen::Tensor<double, 2> one_minus_sigmoid = sigmoid_cache_2d_.unaryExpr([](double x) {
                return 1.0 - x;
            });
            return da * sigmoid_cache_2d_ * one_minus_sigmoid;
        }

        // scalar version
        double Sigmoid::forward(double z)
        {
            return 1.0 / (1.0 + std::exp(-z));
        }

        // 4D tensor version
        Eigen::Tensor<double, 4> Sigmoid::forward(const Eigen::Tensor<double, 4>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("Empty input tensor!");
            }
            in_cache_4d_ = z;
            
            sigmoid_cache_4d_ = z.unaryExpr([](double x) {
                return 1.0 / (1.0 + std::exp(-x));
            });
            return sigmoid_cache_4d_;
        }

        Eigen::Tensor<double, 4> Sigmoid::backward(const Eigen::Tensor<double, 4>& da)
        {
            if (da.dimension(0) != in_cache_4d_.dimension(0) ||
                da.dimension(1) != in_cache_4d_.dimension(1) ||
                da.dimension(2) != in_cache_4d_.dimension(2) ||
                da.dimension(3) != in_cache_4d_.dimension(3) ||
                da.size() == 0)
            {
                throw std::runtime_error("Shape mismatch or empty input: da and in_cache_ must have equal non-zero size!");
            }
            
            Eigen::Tensor<double, 4> one_minus_sigmoid = sigmoid_cache_4d_.unaryExpr([](double x) {
                return 1.0 - x;
            });
            return da * sigmoid_cache_4d_ * one_minus_sigmoid;
        }
    }
}