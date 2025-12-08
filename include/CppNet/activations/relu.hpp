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

                virtual Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_actication) = 0;
                virtual Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) = 0;
                
                virtual Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_actication) = 0;
                virtual Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output) = 0;

                virtual ~Activation() = default;
        };

        class ReLU : public Activation
        {
            public:
                ReLU();
                
                Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_activation);
                Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output);
                
                Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_activation);
                Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output);

                static void set_num_threads(int num_threads);

            private:

                Eigen::Tensor<float, 2> output_cache_2d_; // cache the output for backward pass
                Eigen::Tensor<float, 4> output_cache_4d_; 
        };
    }
}

#endif