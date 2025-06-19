#include "activations.hpp"
#include <cmath>

namespace CppNet
{
    namespace Activations
    {
        /************************Rectifed linear unit (ReLU)************************/
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

        /************************Sigmoid************************/
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
        /************************SoftMax************************/

        Eigen::Tensor<double, 2> SoftMax::forward(const Eigen::Tensor<double, 2>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("SoftMax: Empty input tensor in 2D forward");
            }

            in_cache_2d_ = z;

            // Subtract max for numerical stability
            Eigen::Tensor<double, 1> max_z = z.maximum(Eigen::array<int, 1>{1});
            Eigen::Tensor<double, 2> max_z_reshaped = max_z.reshape(Eigen::array<int, 2>{z.dimension(0), 1});
            Eigen::Tensor<double, 2> z_stable = z - max_z_reshaped.broadcast(Eigen::array<int, 2>{1, z.dimension(1)});

            Eigen::Tensor<double, 2> exp_z = z_stable.exp();
            Eigen::Tensor<double, 1> sum_exp_z = exp_z.sum(Eigen::array<int, 1>{1});
            Eigen::Tensor<double, 2> sum_exp_z_reshaped = sum_exp_z.reshape(Eigen::array<int, 2>{z.dimension(0), 1});
            softmax_cache_2d_ = exp_z / (
            sum_exp_z_reshaped.broadcast(Eigen::array<int, 2>{1, static_cast<int>(z.dimension(1))}) + 1e-15);
            //softmax_cache_2d_ = exp_z / (sum_exp_z_reshaped.broadcast(Eigen::array<int, 2>{1, z.dimension(1)}) + 1e-15);
            return softmax_cache_2d_;
        }

        Eigen::Tensor<double, 2> SoftMax::backward(const Eigen::Tensor<double, 2>& da)
        {
            if (da.dimension(0) != in_cache_2d_.dimension(0) ||
                da.dimension(1) != in_cache_2d_.dimension(1) ||
                da.size() == 0)
            {
                throw std::runtime_error("SoftMax: Shape mismatch or empty input in 2D backward");
            }

            // Gradient of SoftMax: dz = da * softmax - softmax * sum(da * softmax)
            Eigen::Tensor<double, 2> dz = softmax_cache_2d_ * da;
            Eigen::Tensor<double, 1> sum_da_softmax = dz.sum(Eigen::array<int, 1>{1});
            Eigen::Tensor<double, 2> sum_da_softmax_reshaped = sum_da_softmax.reshape(Eigen::array<int, 2>{da.dimension(0), 1});
            dz = dz - softmax_cache_2d_ * sum_da_softmax_reshaped.broadcast(Eigen::array<int, 2>{1, static_cast<int>(da.dimension(1))});
            return dz;
        }

        double SoftMax::forward(double z)
        {
            // SoftMax on a single value is undefined; return z for consistency
            return z;
        }

        Eigen::Tensor<double, 4> SoftMax::forward(const Eigen::Tensor<double, 4>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("SoftMax: Empty input tensor in 4D forward");
            }

            in_cache_4d_ = z;

            // Subtract max over channel dimension (dim 1) for numerical stability
            Eigen::Tensor<double, 3> max_z = z.maximum(Eigen::array<int, 1>{1});
            Eigen::Tensor<double, 4> max_z_reshaped = max_z.reshape(Eigen::array<int, 4>{
                z.dimension(0), 1, z.dimension(2), z.dimension(3)
            });

            Eigen::Tensor<double, 4> z_stable = z - max_z_reshaped.broadcast(Eigen::array<int, 4>{
                1, z.dimension(1), 1, 1
            });

            Eigen::Tensor<double, 4> exp_z = z_stable.exp();
            Eigen::Tensor<double, 3> sum_exp_z = exp_z.sum(Eigen::array<int, 1>{1});
            Eigen::Tensor<double, 4> sum_exp_z_reshaped = sum_exp_z.reshape(Eigen::array<int, 4>{
                z.dimension(0), 1, z.dimension(2), z.dimension(3)
            });

            softmax_cache_4d_ = exp_z / (sum_exp_z_reshaped.broadcast(Eigen::array<int, 4>{1, static_cast<int>(z.dimension(1)), 1, 1}) + 1e-15);
            return softmax_cache_4d_;
        }

        Eigen::Tensor<double, 4> SoftMax::backward(const Eigen::Tensor<double, 4>& da)
        {
            if (da.dimension(0) != in_cache_4d_.dimension(0) ||
                da.dimension(1) != in_cache_4d_.dimension(1) ||
                da.dimension(2) != in_cache_4d_.dimension(2) ||
                da.dimension(3) != in_cache_4d_.dimension(3) ||
                da.size() == 0)
            {
                throw std::runtime_error("SoftMax: Shape mismatch or empty input in 4D backward");
            }

            Eigen::Tensor<double, 4> dz = softmax_cache_4d_ * da;
            Eigen::Tensor<double, 3> sum_da_softmax = dz.sum(Eigen::array<int, 1>{1});
            Eigen::Tensor<double, 4> sum_da_softmax_reshaped = sum_da_softmax.reshape(Eigen::array<int, 4>{
                da.dimension(0), 1, da.dimension(2), da.dimension(3)
            });

            dz = dz - softmax_cache_4d_ * sum_da_softmax_reshaped.broadcast(Eigen::array<int, 4>{1, static_cast<int>(da.dimension(1)), 1, 1});

            return dz;
        }

    }
}