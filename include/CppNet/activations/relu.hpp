#ifndef RELU_HPP
#define RELU_HPP

#include "CppNet/activations/activation.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

namespace CppNet
{
namespace Activations
    {
        class ReLU : public Activation
        {
        public:
            ReLU();
            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_activation) override;
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) override;
            Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_activation) override;
            Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output) override;
            static void set_num_threads(int num_threads);
            
        private:
            Eigen::Tensor<float, 2> output_cache_2d_;
            Eigen::Tensor<float, 4> output_cache_4d_;
             
        };
    }
}

#endif // RELU_HPP