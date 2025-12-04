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

                virtual Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& pre_actication) = 0;
                virtual Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output) = 0;
                
                virtual Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& pre_actication) = 0;
                virtual Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& grad_output) = 0;

                virtual ~Activation() = default;
        };

        class Sigmoid : public Activation
        {
            public:

                Sigmoid();
                
                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& pre_activation);
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output);
                
                Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& pre_activation);
                Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& grad_output);

                static void set_num_threads(int num_threads);

            private:

                Eigen::Tensor<double, 2> input_cache_2d_; 
                Eigen::Tensor<double, 2> output_cache_2d_;
                Eigen::Tensor<double, 4> input_cache_4d_;
                Eigen::Tensor<double, 4> output_cache_4d_;
        };   
    }
}

#endif 