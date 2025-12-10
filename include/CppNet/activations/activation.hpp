#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace CppNet
{
namespace Activations
    {
        class Activation
        {
        public:
            virtual Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_activation) = 0;
            virtual Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) = 0;
            virtual Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_activation) = 0;
            virtual Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output) = 0;
            virtual ~Activation() = default;
        };
    }
}

#endif // ACTIVATION_HPP