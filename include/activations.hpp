#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <vector>

namespace CppNet
{
    namespace Activations
    {
        class Activation
        {
        public:
            virtual ~Activation() = default;
        };

        class Sigmoid : public Activation
        {
        public:
            Sigmoid();
            
            // Support different tensor ranks for flexibility
            Eigen::Tensor<double, 1> forward(const Eigen::Tensor<double, 1>& input);
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input);
            
            Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& input);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input);
            
            Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input);
        };

        class Tanh : public Activation
        {
        public:
            Tanh();
            
            Eigen::Tensor<double, 1> forward(const Eigen::Tensor<double, 1>& input);
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input);
            
            Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& input);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input);
            
            Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input);
        };

        class ReLU : public Activation
        {
        public:
            ReLU();
            
            Eigen::Tensor<double, 1> forward(const Eigen::Tensor<double, 1>& input);
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input);
            
            Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& input);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input);
            
            Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input);
        };

        class LeakyReLU : public Activation
        {
        private:
            double negative_slope;
            
        public:
            LeakyReLU(double negative_slope = 0.01);
            
            Eigen::Tensor<double, 1> forward(const Eigen::Tensor<double, 1>& input);
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input);
            
            Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& input);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input);
            
            Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input);
        };

        class Softmax : public Activation
        {
        private:
            int axis; // Axis along which to apply softmax
            
        public:
            Softmax(int axis = -1); // -1 means last axis
            
            Eigen::Tensor<double, 1> forward(const Eigen::Tensor<double, 1>& input);
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input);
            
            Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& input);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input);
            
            Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input);
        };
    }
}

#endif // ACTIVATIONS_HPP