#include <cmath>
#include <Eigen/Dense>
#include "activations.hpp"

namespace CppNet
{
    namespace Activations
    {
        /************************************** Sigmoid *************************************/
        Sigmoid::Sigmoid() 
        {
            // No parameters needed for Sigmoid
        }

        Eigen::Tensor<double, 1> Sigmoid::forward(const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement sigmoid: 1 / (1 + exp(-x))
            Eigen::Tensor<double, 1> output(input.dimension(0));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 1> Sigmoid::backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
            Eigen::Tensor<double, 1> grad_input(input.dimension(0));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 2> Sigmoid::forward(const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> output(input.dimension(0), input.dimension(1));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 2> Sigmoid::backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> grad_input(input.dimension(0), input.dimension(1));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 3> Sigmoid::forward(const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 3> Sigmoid::backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> grad_input(input.dimension(0), input.dimension(1), input.dimension(2));
            grad_input.setZero();
            return grad_input;
        }

        /************************************** Tanh *************************************/
        Tanh::Tanh() 
        {
            // No parameters needed for Tanh
        }

        Eigen::Tensor<double, 1> Tanh::forward(const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            Eigen::Tensor<double, 1> output(input.dimension(0));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 1> Tanh::backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement tanh derivative: 1 - tanh²(x)
            Eigen::Tensor<double, 1> grad_input(input.dimension(0));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 2> Tanh::forward(const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> output(input.dimension(0), input.dimension(1));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 2> Tanh::backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> grad_input(input.dimension(0), input.dimension(1));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 3> Tanh::forward(const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 3> Tanh::backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> grad_input(input.dimension(0), input.dimension(1), input.dimension(2));
            grad_input.setZero();
            return grad_input;
        }

        /************************************** ReLU *************************************/
        ReLU::ReLU() 
        {
            // No parameters needed for ReLU
        }

        Eigen::Tensor<double, 1> ReLU::forward(const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement ReLU: max(0, x)
            Eigen::Tensor<double, 1> output(input.dimension(0));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 1> ReLU::backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement ReLU derivative: 1 if x > 0, else 0
            Eigen::Tensor<double, 1> grad_input(input.dimension(0));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 2> ReLU::forward(const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> output(input.dimension(0), input.dimension(1));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 2> ReLU::backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> grad_input(input.dimension(0), input.dimension(1));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 3> ReLU::forward(const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 3> ReLU::backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> grad_input(input.dimension(0), input.dimension(1), input.dimension(2));
            grad_input.setZero();
            return grad_input;
        }

        /************************************** LeakyReLU *************************************/
        LeakyReLU::LeakyReLU(double negative_slope) : negative_slope(negative_slope) 
        {
            // Store the negative slope parameter
        }

        Eigen::Tensor<double, 1> LeakyReLU::forward(const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement LeakyReLU: x if x > 0, else negative_slope * x
            Eigen::Tensor<double, 1> output(input.dimension(0));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 1> LeakyReLU::backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement LeakyReLU derivative: 1 if x > 0, else negative_slope
            Eigen::Tensor<double, 1> grad_input(input.dimension(0));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 2> LeakyReLU::forward(const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> output(input.dimension(0), input.dimension(1));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 2> LeakyReLU::backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> grad_input(input.dimension(0), input.dimension(1));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 3> LeakyReLU::forward(const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 3> LeakyReLU::backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> grad_input(input.dimension(0), input.dimension(1), input.dimension(2));
            grad_input.setZero();
            return grad_input;
        }

        /************************************** Softmax *************************************/
        Softmax::Softmax(int axis) : axis(axis) 
        {
            // Store the axis parameter
        }

        Eigen::Tensor<double, 1> Softmax::forward(const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement Softmax: exp(x_i) / sum(exp(x_j)) for all j
            // Remember to subtract max for numerical stability
            Eigen::Tensor<double, 1> output(input.dimension(0));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 1> Softmax::backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement Softmax derivative: s_i * (δ_ij - s_j) where s = softmax(x)
            Eigen::Tensor<double, 1> grad_input(input.dimension(0));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 2> Softmax::forward(const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> output(input.dimension(0), input.dimension(1));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 2> Softmax::backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> grad_input(input.dimension(0), input.dimension(1));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 3> Softmax::forward(const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 3> Softmax::backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> grad_input(input.dimension(0), input.dimension(1), input.dimension(2));
            grad_input.setZero();
            return grad_input;
        }
    }
}