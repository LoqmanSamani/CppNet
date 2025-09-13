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
                //virtual double forward(double z) = 0;
                //virtual Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& z) = 0;
                //virtual Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& da) = 0;
                virtual ~Activation() = default;

            //protected:

            //    Eigen::Tensor<double, 2> in_cache_2d_;
            //    Eigen::Tensor<double, 4> in_cache_4d_;
            //    Eigen::Tensor<double, 2> sigmoid_cache_2d_; 
            //    Eigen::Tensor<double, 4> sigmoid_cache_4d_; 
            //    Eigen::Tensor<double, 2> softmax_cache_2d_; 
            //    Eigen::Tensor<double, 4> softmax_cache_4d_;
        };

        class Sigmoid : public Activation
        {
            public:
                Sigmoid();
                
                // Support different tensor ranks for flexibility
                //Eigen::Tensor<double, 1> forward(const Eigen::Tensor<double, 1>& input);
                //Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input);
                
                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& pre_activation);
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output);
                
                //Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input);
                //Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input);
            private:
                Eigen::Tensor<double, 2> input_cache_2d_; // Cache the input for backward pass
                Eigen::Tensor<double, 2> output_cache_2d_; // Cache the output for backward pass
                //Eigen::Tensor<double, 1> output_cache_1d_;
                //Eigen::Tensor<double, 3> output_cache_3d_;
        };

        class ReLU : public Activation
        {
            public:
                ReLU();
                
                //Eigen::Tensor<double, 1> forward(const Eigen::Tensor<double, 1>& input);
                //Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input);
                
                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& pre_activation);
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output);
                
                //Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input);
                //Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input);
            private:
                Eigen::Tensor<double, 2> output_cache_2d_; // Cache the output for backward pass
                //Eigen::Tensor<double, 1> output_cache_1d_;
                //Eigen::Tensor<double, 3> output_cache_3d_;   
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