#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "layers.hpp"




namespace CppNet
{
    namespace Activations
    {
        class ReLU : public Layers::Layer
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

        class Sigmoid : public Layers::Layer
        {
            public:

                // compute sigmoid activation: 1 / (1 + exp(-z))
                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z);

                // compute gradient of sigmoid
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da);

                bool is_trainable() const override { return false; }
                void update_parameters(Optimizer&, double) override {}

            private:

                Eigen::Tensor<double, 2> in_cache_;
                Eigen::Tensor<double, 2> sigmoid_cache_;
        };
    }
}




#endif // ACTIVATIONS_HPP