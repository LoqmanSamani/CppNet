#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace CppNet
{
    namespace Layers
    {
        class Layer
        {
        public:
            virtual ~Layer() = default;
        };

        class Linear : public Layer
        {
        public:
            Linear();
            Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& input);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output);
        };

        class Conv2d : public Layer
        {
        public:
            Conv2d();
            Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& input);
            Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& grad_output);
        };

        class MaxPool2D : public Layer
        {
        public:
            MaxPool2D();
            Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& input);
            Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& grad_output);
        };

        class Flatten : public Layer
        {
        public:
            Flatten();
            Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 4>& input); 
            Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 2>& grad_output);
        };

        class MultiHeadAttention : public Layer
        {
        public:
            MultiHeadAttention();
            Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output);
        };

        class RNN : public Layer
        {
        public:
            RNN();
            Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input); 
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output);
        };
    }
}

#endif // LAYERS_HPP