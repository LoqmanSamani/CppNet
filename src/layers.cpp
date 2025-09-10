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
            double scale = 0.0;
            double mean = 0.0;
            double std_dev = 0.0;
            bool use_normal = false;  // Flag to determine distribution type
            
            // Calculate initialization parameters based on method
            if (weight_init_ == "xavier" || weight_init_ == "xavier_uniform")
            {
                // Xavier/Glorot Uniform: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
                scale = std::sqrt(6.0 / (in_size_ + out_size_));
                use_normal = false;
            }
            else if (weight_init_ == "xavier_normal" || weight_init_ == "glorot_normal")
            {
                // Xavier/Glorot Normal: N(0, sqrt(2/(fan_in + fan_out)))
                mean = 0.0;
                std_dev = std::sqrt(2.0 / (in_size_ + out_size_));
                use_normal = true;
            }
            else if (weight_init_ == "he" || weight_init_ == "he_uniform")
            {
                // He Uniform (for ReLU): U(-sqrt(6/fan_in), sqrt(6/fan_in))
                scale = std::sqrt(6.0 / in_size_);
                use_normal = false;
            }
            else if (weight_init_ == "he_normal")
            {
                // He Normal (for ReLU): N(0, sqrt(2/fan_in))
                mean = 0.0;
                std_dev = std::sqrt(2.0 / in_size_);
                use_normal = true;
            }
            else if (weight_init_ == "lecun_uniform")
            {
                // LeCun Uniform: U(-sqrt(3/fan_in), sqrt(3/fan_in))
                scale = std::sqrt(3.0 / in_size_);
                use_normal = false;
            }
            else if (weight_init_ == "lecun_normal")
            {
                // LeCun Normal: N(0, sqrt(1/fan_in))
                mean = 0.0;
                std_dev = std::sqrt(1.0 / in_size_);
                use_normal = true;
            }
            else if (weight_init_ == "uniform")
            {
                // Simple uniform distribution: U(-0.1, 0.1)
                scale = 0.1;
                use_normal = false;
            }
            else if (weight_init_ == "normal")
            {
                // Simple normal distribution: N(0, 0.01)
                mean = 0.0;
                std_dev = 0.01;
                use_normal = true;
            }
            else if (weight_init_ == "zeros")
            {
                // Initialize with zeros (useful for some specific architectures)
                scale = 0.0;
                use_normal = false;
            }
            else if (weight_init_ == "ones")
            {
                // Initialize with ones (rarely used, but available)
                scale = -1.0; // Special flag for ones initialization
                use_normal = false;
            }
            else
            {
                throw std::runtime_error("Unknown weight initialization method: '" + weight_init_ + 
                                        "' in layer: " + layer_name_ + 
                                        "\nSupported methods: xavier, xavier_normal, he, he_normal, " +
                                        "lecun_uniform, lecun_normal, uniform, normal, zeros, ones");
            }

            // Initialize weight tensor
            weights_ = Eigen::Tensor<double, 2>(in_size_, out_size_);

            // Only parallelize for larger matrices to avoid overhead
            const int total_elements = in_size_ * out_size_;
            const bool should_parallelize = (total_elements > 10000);

            if (weight_init_ == "zeros")
            {
                // Special case: zero initialization
                weights_.setZero();
            }
            else if (weight_init_ == "ones")
            {
                // Special case: ones initialization
                weights_.setConstant(1.0);
            }
            else if (should_parallelize)
            {
                // Parallel initialization for large matrices
                #pragma omp parallel
                {
                    // Each thread gets its own random generator to avoid race conditions
                    std::mt19937 local_gen(rd() + omp_get_thread_num() * 1000 + 
                                        std::chrono::high_resolution_clock::now().time_since_epoch().count() % 1000);
                    
                    if (use_normal)
                    {
                        std::normal_distribution<double> local_dist(mean, std_dev);
                        #pragma omp for collapse(2)
                        for (int i = 0; i < in_size_; ++i)
                        {
                            for (int j = 0; j < out_size_; ++j)
                            {
                                weights_(i, j) = local_dist(local_gen);
                            }
                        }
                    }
                    else
                    {
                        std::uniform_real_distribution<double> local_dist(-scale, scale);
                        #pragma omp for collapse(2)
                        for (int i = 0; i < in_size_; ++i)
                        {
                            for (int j = 0; j < out_size_; ++j)
                            {
                                weights_(i, j) = local_dist(local_gen);
                            }
                        }
                    }
                }
            }
            else
            {
                // Serial initialization for small matrices
                std::mt19937 gen(rd());
                
                if (use_normal)
                {
                    std::normal_distribution<double> dist(mean, std_dev);
                    for (int i = 0; i < in_size_; ++i)
                    {
                        for (int j = 0; j < out_size_; ++j)
                        {
                            weights_(i, j) = dist(gen);
                        }
                    }
                }
                else
                {
                    std::uniform_real_distribution<double> dist(-scale, scale);
                    for (int i = 0; i < in_size_; ++i)
                    {
                        for (int j = 0; j < out_size_; ++j)
                        {
                            weights_(i, j) = dist(gen);
                        }
                    }
                }
            }

            // Initialize weight gradients with zeros
            grad_weights_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            grad_weights_.setZero();

            // Initialize biases and bias gradients
            if (bias_)
            {
                biases_ = Eigen::Tensor<double, 1>(out_size_);
                grad_biases_ = Eigen::Tensor<double, 1>(out_size_);
                
                // Biases are typically initialized to zero regardless of weight initialization
                biases_.setZero();
                grad_biases_.setZero();
            }
            else
            {
                // Initialize empty tensors when bias is disabled
                biases_ = Eigen::Tensor<double, 1>(0);
                grad_biases_ = Eigen::Tensor<double, 1>(0);
            }
        }

        void Linear::reinitialize_weights(const std::string& new_init_method)
        {
            std::string old_method = weight_init_;
            weight_init_ = new_init_method;
            
            try
            {
                init_params_and_grads();
                std::cout << "Layer '" << layer_name_ << "' weights reinitialized from '" 
                        << old_method << "' to '" << new_init_method << "'" << std::endl;
            }
            catch (const std::exception& e)
            {
                // Restore old method if new one fails
                weight_init_ = old_method;
                throw std::runtime_error("Failed to reinitialize with method '" + new_init_method + 
                                        "': " + e.what());
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