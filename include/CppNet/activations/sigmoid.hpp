#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "CppNet/activations/activation.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

// header file for Sigmoid activation function implementation

namespace CppNet
{
namespace Activations
    {
        class Sigmoid : public Activation
        {
        public:
            Sigmoid();
            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_activation) override;
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) override;
            Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_activation) override;
            Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output) override;
            static void set_num_threads(int num_threads);
            
        private:
            Eigen::Tensor<float, 2> input_cache_2d_; 
            Eigen::Tensor<float, 2> output_cache_2d_;
            Eigen::Tensor<float, 4> input_cache_4d_;
            Eigen::Tensor<float, 4> output_cache_4d_;
        };   
    }
}

#endif // SIGMOID_HPP