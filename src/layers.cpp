#include <cmath>
#include <omp.h> // For potential parallelization
#include <Eigen/Dense>
#include "layers.hpp"
#include "optimizers.hpp"  // Comment out until implemented
#include "activations.hpp" // Comment out until implemented

namespace CppNet
{
    namespace Layers
    {
        /************************************** Linear/Dense *************************************/
        Linear::Linear(
            int in_size, 
            int out_size, 
            std::string layer_name, 
            bool trainable, 
            bool bias, 
            std::string device,
            std::string weight_init
        ) 
        : in_size_(in_size), out_size_(out_size), layer_name_(layer_name), 
          trainable_(trainable), bias_(bias), device_(device), weight_init_(weight_init)
        {
            // check if in and out sizes are positive integers
            if (in_size <= 0 || out_size <= 0)
            {
                throw std::runtime_error("in_size and out_size of layer: " + layer_name + " must be positive integers!");
            }

            if (layer_name.empty()) 
            {
            layer_name_ = "Linear_" + std::to_string(in_size) + "x" + std::to_string(out_size);
            }

            // initialize parameters and gradients
            init_params_and_grads();
        }
        void Linear::init_params_and_grads()
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            double scale;
            
            // scaling factor for Xavier initialization
            if (weight_init_ == "xavier")
            {
                double scale = std::sqrt(6.0 / (in_size_ + out_size_));    
            }
            else if (weight_init_ == "he")
            {
                // for now He is the same as Xavier it will be updated later
                double scale = std::sqrt(6.0 / (in_size_ + out_size_));   
            }
            else
            {
                throw std::runtime_error("Unknown weight initialization method: " + weight_init_ + " in layer: " + layer_name_);
            } 
            std::uniform_real_distribution<> dis(-scale, scale);  
            
            // initialize weight-tensor
            weights_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
           
            // Parallelize weight initialization
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < in_size_; ++i) 
            {
                for (int j = 0; j < out_size_; ++j) 
                {
                    // Each thread needs its own random generator
                    thread_local std::mt19937 local_gen(rd() + omp_get_thread_num());
                    thread_local std::uniform_real_distribution<> local_dis(-scale, scale);
                    weights_(i, j) = local_dis(local_gen);
                }
            }

            // initialize weight-gradient matrix with zero
            grad_weights_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            grad_weights_.setZero();

            if (bias_)
            {
                // initialize biases with zero, if bias_ is true.
                biases_ = Eigen::Tensor<double, 1>(out_size_).setZero(); 
                // initialize bias-gradient matrix with zero 
                grad_biases_ = Eigen::Tensor<double, 1>(out_size_).setZero(); 
            }
            else
            {
                // initialize empty biases and gradients when bias is false
                biases_ = Eigen::Tensor<double, 1>(0);
                grad_biases_ = Eigen::Tensor<double, 1>(0);
            }
        }

        void Linear::update_parameters(Optimizers::Optimizer& optimizer, double learning_rate)
        {    
            // commented out until implemented
            // optimizer.update(*this, learning_rate);
        }

        Eigen::Tensor<double, 2> Linear::forward(const Eigen::Tensor<double, 2>& input) 
        {
            // check dimensions
            if (input.dimension(1) != weights_.dimension(0))
            {
                throw std::runtime_error("Shape mismatch: in layer: " + layer_name_ + " input.dimension(1) must be equal weights_.dimension(0)!");
            }
            // store input to use later in gradient calculation
            in_cache_ = input;

            // get dimensions
            const int batch_size = input.dimension(0);
            const int input_size = input.dimension(1);
            const int output_size = weights_.dimension(1);

            // Create output tensor
            Eigen::Tensor<double, 2> output(batch_size, output_size);

            // Manual matrix multiplication without OpenMP
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < batch_size; ++b)
            {
                for (int j = 0; j < output_size; ++j) 
                {
                    double sum = 0.0;
                    // Vectorize inner loop and use reduction for better performance
                    #pragma omp simd reduction(+:sum)
                    for (int i = 0; i < input_size; ++i) 
                    {
                        sum += input(b, i) * weights_(i, j);
                    }
                    output(b, j) = sum;
                }
            }

            // Add bias if enabled
            if (bias_) 
            {
                #pragma omp parallel for collapse(2)
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int j = 0; j < output_size; ++j) 
                    {
                        output(b, j) += biases_(j);
                    }
                }
            }
            
            return output;
        }

        Eigen::Tensor<double, 2> Linear::backward(const Eigen::Tensor<double, 2>& grad_output) 
        {
            // get dimensions
            const int batch_size = grad_output.dimension(0);
            const int output_size = grad_output.dimension(1);
            const int input_size = in_cache_.dimension(1);


            // dimension validation
            if (grad_output.dimension(0) != in_cache_.dimension(0)) 
            {
                throw std::runtime_error("Batch size mismatch in layer: " + layer_name_);
            }
            if (grad_output.dimension(1) != out_size_) 
            {
                throw std::runtime_error("Output size mismatch in layer: " + layer_name_);
            }
            
            // compute parameter gradients (only if trainable)
            if (trainable_) 
            {
                // gradient w.r.t. weights: in_cache_^T * grad_output
                // Manual computation: grad_weights_(i,j) = sum_b(in_cache_(b,i) * grad_output(b,j))
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < input_size; ++i) 
                {
                    for (int j = 0; j < output_size; ++j) 
                    {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (int b = 0; b < batch_size; ++b) 
                        {
                            sum += in_cache_(b, i) * grad_output(b, j);
                        }
                        grad_weights_(i, j) = sum;
                    }
                }
                
                // gradient w.r.t. biases: sum over batch dimension
                if (bias_) 
                {
                    #pragma omp parallel for
                    for (int j = 0; j < output_size; ++j) 
                    {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (int b = 0; b < batch_size; ++b) 
                        {
                            sum += grad_output(b, j);
                        }
                        grad_biases_(j) = sum;
                    }
                }
            }
            
            // compute gradient w.r.t. input: grad_out * W^T
            // Manual computation: grad_input(b,i) = sum_j(grad_output(b,j) * weights_(i,j))
            Eigen::Tensor<double, 2> grad_input(batch_size, input_size);
            
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < batch_size; ++b) 
            {
                for (int i = 0; i < input_size; ++i) 
                {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (int j = 0; j < output_size; ++j) 
                    {
                        sum += grad_output(b, j) * weights_(i, j);
                    }
                    grad_input(b, i) = sum;
                }
            }
            
            return grad_input;
        }
        // helper method to set number of threads
        void Linear::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }

        /************************************** Conv2d *******************************************/
        Conv2d::Conv2d() 
        {
            // TODO: Initialize kernels and biases
        }

        Eigen::Tensor<double, 4> Conv2d::forward(const Eigen::Tensor<double, 4>& input) 
        {
            // TODO: Implement convolution
            // For now, return a tensor with same dimensions as input
            Eigen::Tensor<double, 4> output(input.dimension(0), input.dimension(1), input.dimension(2), input.dimension(3));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 4> Conv2d::backward(const Eigen::Tensor<double, 4>& grad_output) 
        {
            // TODO: Implement convolution backward pass
            Eigen::Tensor<double, 4> grad_input(grad_output.dimension(0), grad_output.dimension(1), grad_output.dimension(2), grad_output.dimension(3));
            grad_input.setZero();
            return grad_input;
        }

        /************************************** MaxPool2D *******************************************/
        MaxPool2D::MaxPool2D() 
        {
            // TODO: Initialize pooling parameters
        }

        Eigen::Tensor<double, 4> MaxPool2D::forward(const Eigen::Tensor<double, 4>& input) 
        {
            // TODO: Implement max pooling
            Eigen::Tensor<double, 4> output(input.dimension(0), input.dimension(1), input.dimension(2)/2, input.dimension(3)/2);
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 4> MaxPool2D::backward(const Eigen::Tensor<double, 4>& grad_output) 
        {
            // TODO: Implement max pooling backward pass
            Eigen::Tensor<double, 4> grad_input(grad_output.dimension(0), grad_output.dimension(1), grad_output.dimension(2)*2, grad_output.dimension(3)*2);
            grad_input.setZero();
            return grad_input;
        }

        /*************************************** Flatten ********************************************/
        Flatten::Flatten() 
        {
            // No parameters needed for flatten
        }

        Eigen::Tensor<double, 2> Flatten::forward(const Eigen::Tensor<double, 4>& input) 
        {
            // TODO: Implement flattening (4D -> 2D)
            int batch_size = input.dimension(0);
            int flattened_size = input.dimension(1) * input.dimension(2) * input.dimension(3);
            Eigen::Tensor<double, 2> output(batch_size, flattened_size);
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 4> Flatten::backward(const Eigen::Tensor<double, 2>& grad_output) 
        {
            // TODO: Implement unflatten for backward pass (2D -> 4D)
            // You'll need to store original dimensions from forward pass
            int batch_size = grad_output.dimension(0);
            Eigen::Tensor<double, 4> grad_input(batch_size, 1, 1, grad_output.dimension(1)); // Placeholder dimensions
            grad_input.setZero();
            return grad_input;
        }

        /********************************* Multi-Head Attention *************************************/
        MultiHeadAttention::MultiHeadAttention() 
        {
            // TODO: Initialize attention parameters
        }

        Eigen::Tensor<double, 3> MultiHeadAttention::forward(const Eigen::Tensor<double, 3>& input) 
        {
            // TODO: Implement multi-head attention
            Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 3> MultiHeadAttention::backward(const Eigen::Tensor<double, 3>& grad_output) 
        {
            // TODO: Implement attention backward pass
            Eigen::Tensor<double, 3> grad_input(grad_output.dimension(0), grad_output.dimension(1), grad_output.dimension(2));
            grad_input.setZero();
            return grad_input;
        }

        /************************************** RNN *************************************/
        RNN::RNN() 
        {
            // TODO: Initialize RNN parameters
        }

        Eigen::Tensor<double, 3> RNN::forward(const Eigen::Tensor<double, 3>& input) 
        {
            // TODO: Implement RNN forward pass
            Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 3> RNN::backward(const Eigen::Tensor<double, 3>& grad_output) 
        {
            // TODO: Implement RNN backward pass
            Eigen::Tensor<double, 3> grad_input(grad_output.dimension(0), grad_output.dimension(1), grad_output.dimension(2));
            grad_input.setZero();
            return grad_input;
        }
    }
}