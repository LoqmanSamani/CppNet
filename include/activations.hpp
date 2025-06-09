
#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "layers.hpp"
#include "optimizers.hpp"





namespace CppNet
{
    namespace Activations
    {
        // Base activation class
        class Activation : public Layers::Layer
        {
            public:
                // pure virtual methods for activation functions
                virtual Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z) = 0;
                virtual Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da) = 0;
                
                // activations are never trainable!!!
                bool is_trainable() const override final { return false; }
                void update_parameters(Optimizers::Optimizer&, double) override final {}
                
                virtual ~Activation() = default;
        };

        class ReLU : public Activation
        {
            public:
                // compute ReLU activation: max(0, z)
                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z) override;
                // compute gradient of ReLU
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da) override;
                
            private:
                Eigen::Tensor<double, 2> in_cache_;
        };

        class Sigmoid : public Activation
        {
            public:
                // compute sigmoid activation: 1 / (1 + exp(-z))
                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z) override;
                // compute gradient of sigmoid
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da) override;
                
            private:
                Eigen::Tensor<double, 2> in_cache_;
                Eigen::Tensor<double, 2> sigmoid_cache_;
        };
    }
}



#endif // ACTIVATIONS_HPP