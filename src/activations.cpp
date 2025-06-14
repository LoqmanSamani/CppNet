#include "activations.hpp"
#include <cmath>

namespace CppNet
{
    namespace Activations
    {
        // ReLU implementations
        Eigen::Tensor<double, 2> ReLU::forward(const Eigen::Tensor<double, 2>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("ReLU: Empty input tensor in 2D forward");
            }
            in_cache_2d_ = z;
            return z.cwiseMax(0.0);
        }

        Eigen::Tensor<double, 2> ReLU::backward(const Eigen::Tensor<double, 2>& da)
        {
            if (da.dimension(0) != in_cache_2d_.dimension(0) ||
                da.dimension(1) != in_cache_2d_.dimension(1) ||
                da.size() == 0)
            {
                throw std::runtime_error("ReLU: Shape mismatch or empty input in 2D backward");
            }
            
            Eigen::Tensor<double, 2> mask = (in_cache_2d_ > 0.0).template cast<double>();
            return da * mask;
        }

        double ReLU::forward(double z)
        {
            return std::max(0.0, z);
        }

        Eigen::Tensor<double, 4> ReLU::forward(const Eigen::Tensor<double, 4>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("ReLU: Empty input tensor in 4D forward");
            }
            in_cache_4d_ = z;
            return z.cwiseMax(0.0);
        }

        Eigen::Tensor<double, 4> ReLU::backward(const Eigen::Tensor<double, 4>& da)
        {
            if (da.dimension(0) != in_cache_4d_.dimension(0) ||
                da.dimension(1) != in_cache_4d_.dimension(1) ||
                da.dimension(2) != in_cache_4d_.dimension(2) ||
                da.dimension(3) != in_cache_4d_.dimension(3) ||
                da.size() == 0)
            {
                throw std::runtime_error("ReLU: Shape mismatch or empty input in 4D backward");
            }
            
            Eigen::Tensor<double, 4> mask = (in_cache_4d_ > 0.0).template cast<double>();
            return da * mask;
        }

        // Sigmoid implementations
        Eigen::Tensor<double, 2> Sigmoid::forward(const Eigen::Tensor<double, 2>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Empty input tensor in 2D forward");
            }
            in_cache_2d_ = z;
            sigmoid_cache_2d_ = (1.0 + (-z).exp()).inverse();
            return sigmoid_cache_2d_;
        }

        Eigen::Tensor<double, 2> Sigmoid::backward(const Eigen::Tensor<double, 2>& da)
        {
            if (da.dimension(0) != in_cache_2d_.dimension(0) ||
                da.dimension(1) != in_cache_2d_.dimension(1) ||
                da.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Shape mismatch or empty input in 2D backward");
            }
            
            return da * sigmoid_cache_2d_ * (1.0 - sigmoid_cache_2d_);
        }

        double Sigmoid::forward(double z)
        {
            double exp_neg_z = std::exp(-z);
            return 1.0 / (1.0 + exp_neg_z);
        }

        Eigen::Tensor<double, 4> Sigmoid::forward(const Eigen::Tensor<double, 4>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Empty input tensor in 4D forward");
            }
            in_cache_4d_ = z;
            sigmoid_cache_4d_ = (1.0 + (-z).exp()).inverse();
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
                throw std::runtime_error("Sigmoid: Shape mismatch or empty input in 4D backward");
            }
            
            return da * sigmoid_cache_4d_ * (1.0 - sigmoid_cache_4d_);
        }
    }
}