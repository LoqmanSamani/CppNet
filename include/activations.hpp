#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <Eigen/Dense>
#include "layers.hpp"



namespace CppNet
{
    class ReLU : public Layer
    {
    public:
        // compute ReLU activation: max(0, z)
        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z);
        // compute gradient of ReLU
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da);

        bool is_trainable() const override { return false; }
        void update_parameters(Optimizer&, double) override {}

    private:
        Eigen::Tensor<double, 2> in_cache_;
    };

    class Sigmoid : public Layer
    {
    public:
        // compute sigmoid activation: 1 / (1 + exp(-z))
        Eigen::MatrixXd forward(const Eigen::MatrixXd& z);
        // compute gradient of sigmoid
        Eigen::MatrixXd backward(const Eigen::MatrixXd& da);

        bool is_trainable() const override { return false; }
        void update_parameters(Optimizer&, double) override {}

    private:
        Eigen::MatrixXd in_cache_;
    };
}

#endif // ACTIVATIONS_HPP