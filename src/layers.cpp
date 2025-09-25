#include <cmath>
#include <omp.h> 
#include <Eigen/Dense>
#include "layers.hpp"
#include "optimizers.hpp"  
#include "activations.hpp" 

namespace CppNet
{
    namespace Layers
    {
        //********************* Linear (Fully Connected: Dense) Layer *********************//
        Linear::Linear(
            int in_size, 
            int out_size, 
            std::string layer_name, 
            bool trainable, 
            bool bias, 
            std::string device,
            std::string weight_init,
            int parallel_threshold
        ) 
        : in_size_(in_size), out_size_(out_size), layer_name_(layer_name), 
          trainable_(trainable), bias_(bias), device_(device), weight_init_(weight_init),
          parallel_threshold_(parallel_threshold)
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
                throw std::runtime_error("Failed to reinitialize with method '" + new_init_method + "': " + e.what());
            }
        }
        void Linear::step(Optimizers::Optimizer& optimizer, double learning_rate)
        {
            optimizer.step(*this, learning_rate);
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

        //********************* Convolutional Layer(2D) *********************//
        Conv2d::Conv2d(
            int in_channels,
            int out_channels,
            std::tuple<int , int> kernel_size,
            std::tuple<int, int> stride,
            std::string layer_name,
            bool trainable,
            bool bias,
            std::string padding,
            std::tuple<int, int, int, int> num_padding,
            std::string padding_mode,  
            std::string device,
            std::string weight_init,
            int parallel_threshold
        ) : in_channels_(in_channels), out_channels_(out_channels),
            kernel_size_(kernel_size), stride_(stride), layer_name_(layer_name),
            trainable_(trainable), bias_(bias), padding_(padding), num_padding_(num_padding),
            padding_mode_(padding_mode), device_(device), weight_init_(weight_init), 
            parallel_threshold_(parallel_threshold)
            //activation_(activation)
        {
            // Input validation
            if (in_channels <= 0 || out_channels <= 0) 
            {
                throw std::runtime_error("in_channels and out_channels must be positive in layer: " + layer_name_);
            }
            if (std::get<0>(kernel_size_) <= 0 || std::get<1>(kernel_size_) <= 0) 
            {
                throw std::runtime_error("Kernel size must be positive in layer: " + layer_name_);
            }
            if (std::get<0>(stride_) <= 0 || std::get<1>(stride_) <= 0) 
            {
                throw std::runtime_error("Stride must be positive in layer: " + layer_name_);
            }
            if (padding_ != "valid" && padding_ != "same" && padding_ != "none") 
            {
                throw std::runtime_error("Invalid padding: " + padding_ + " in layer: " + layer_name_);
            }
            if (padding_mode_ != "zero" && padding_mode_ != "reflect" && padding_mode_ != "edge") 
            {
                throw std::runtime_error("Invalid padding mode: " + padding_mode_ + " in layer: " + layer_name_);
            }
            if (std::get<0>(num_padding_) < 0 || std::get<1>(num_padding_) < 0 || std::get<2>(num_padding_) < 0 || std::get<3>(num_padding_) < 0)
            {
                throw std::runtime_error("Padding values must be non-negative in layer: " + layer_name_);
            }

            // default layer name
            if (layer_name_.empty()) 
            {
                layer_name_ = "Conv2D_" + std::to_string(in_channels_) + "x" + std::to_string(out_channels_);
            }

            // initialize parameters and gradients
            init_params_and_grads();
        }

        void Conv2d::init_params_and_grads()
        {
            std::random_device rd;
            int fan_in = in_channels_ * std::get<0>(kernel_size_) * std::get<1>(kernel_size_);
            int fan_out = out_channels_ * std::get<0>(kernel_size_) * std::get<1>(kernel_size_);
            double scale = 0.0;
            double mean = 0.0;
            double std_dev = 0.0;
            bool use_normal = false;  // Flag to determine distribution type
            
            // Calculate initialization parameters based on method
            if (weight_init_ == "xavier" || weight_init_ == "xavier_uniform")
            {
                // Xavier/Glorot Uniform: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
                scale = std::sqrt(6.0 / (fan_in + fan_out));
                use_normal = false;
            }
            else if (weight_init_ == "xavier_normal" || weight_init_ == "glorot_normal")
            {
                // Xavier/Glorot Normal: N(0, sqrt(2/(fan_in + fan_out)))
                mean = 0.0;
                std_dev = std::sqrt(2.0 / (fan_in + fan_out));
                use_normal = true;
            }
            else if (weight_init_ == "he" || weight_init_ == "he_uniform")
            {
                // He Uniform (for ReLU): U(-sqrt(6/fan_in), sqrt(6/fan_in))
                scale = std::sqrt(6.0 / fan_in);
                use_normal = false;
            }
            else if (weight_init_ == "he_normal")
            {
                // He Normal (for ReLU): N(0, sqrt(2/fan_in))
                mean = 0.0;
                std_dev = std::sqrt(2.0 / fan_in);
                use_normal = true;
            }
            else if (weight_init_ == "lecun_uniform")
            {
                // LeCun Uniform: U(-sqrt(3/fan_in), sqrt(3/fan_in))
                scale = std::sqrt(3.0 / fan_in);
                use_normal = false;
            }
            else if (weight_init_ == "lecun_normal")
            {
                // LeCun Normal: N(0, sqrt(1/fan_in))
                mean = 0.0;
                std_dev = std::sqrt(1.0 / fan_in);
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
            weights_ = Eigen::Tensor<double, 4>(out_channels_, in_channels_, std::get<0>(kernel_size_), std::get<1>(kernel_size_));

            // Only parallelize for larger matrices to avoid overhead
            const int total_elements = out_channels_ * in_channels_ * std::get<0>(kernel_size_) * std::get<1>(kernel_size_);
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
                    // Each thread gets its own random generator
                    std::mt19937 local_gen(rd() + omp_get_thread_num() * 1000 + 
                                        std::chrono::high_resolution_clock::now().time_since_epoch().count() % 1000);
                    if (use_normal)
                    {
                        std::normal_distribution<double> local_dist(mean, std_dev);
                        #pragma omp for collapse(4)
                        for (int oc = 0; oc < out_channels_; ++oc)
                        {
                            for (int ic = 0; ic < in_channels_; ++ic)
                            {
                                for (int kh = 0; kh < std::get<0>(kernel_size_); ++kh)
                                {
                                    for (int kw = 0; kw < std::get<1>(kernel_size_); ++kw)
                                    {
                                        weights_(oc, ic, kh, kw) = local_dist(local_gen);
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        std::uniform_real_distribution<double> local_dist(-scale, scale);
                        #pragma omp for collapse(4)
                        for (int oc = 0; oc < out_channels_; ++oc)
                        {
                            for (int ic = 0; ic < in_channels_; ++ic)
                            {
                                for (int kh = 0; kh < std::get<0>(kernel_size_); ++kh)
                                {
                                    for (int kw = 0; kw < std::get<1>(kernel_size_); ++kw)
                                    {
                                        weights_(oc, ic, kh, kw) = local_dist(local_gen);
                                    }
                                }
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
                    for (int oc = 0; oc < out_channels_; ++oc)
                    {
                        for (int ic = 0; ic < in_channels_; ++ic)
                        {
                            for (int kh = 0; kh < std::get<0>(kernel_size_); ++kh)
                            {
                                for (int kw = 0; kw < std::get<1>(kernel_size_); ++kw)
                                {
                                    weights_(oc, ic, kh, kw) = dist(gen);
                                }
                            }
                        }
                    }
                }
                else
                {
                    std::uniform_real_distribution<double> dist(-scale, scale);
                    for (int oc = 0; oc < out_channels_; ++oc)
                    {
                        for (int ic = 0; ic < in_channels_; ++ic)
                        {
                            for (int kh = 0; kh < std::get<0>(kernel_size_); ++kh)
                            {
                                for (int kw = 0; kw < std::get<1>(kernel_size_); ++kw)
                                {
                                    weights_(oc, ic, kh, kw) = dist(gen);
                                }
                            }
                        }
                    }
                }
            }   
            // Initialize weight gradients with zeros
            grad_weights_ = Eigen::Tensor<double, 4>(out_channels_, in_channels_, std::get<0>(kernel_size_), std::get<1>(kernel_size_));
            grad_weights_.setZero();    
            // Initialize biases and bias gradients
            if (bias_)
            {
                biases_ = Eigen::Tensor<double, 1>(out_channels_);
                grad_biases_ = Eigen::Tensor<double, 1>(out_channels_);     
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

        void Conv2d::reinitialize_weights(const std::string& new_init_method)
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
                throw std::runtime_error("Failed to reinitialize with method '" + new_init_method + "': " + e.what());
            }
        }

        void Conv2d::step(Optimizers::Optimizer& optimizer, double learning_rate)
        {
           optimizer.step(*this, learning_rate);
        }

        Eigen::Tensor<double, 4> Conv2d::pad_input(const Eigen::Tensor<double, 4>& input) 
        {
            int pad_top = std::get<0>(num_padding_);
            int pad_bottom = std::get<1>(num_padding_);
            int pad_left = std::get<2>(num_padding_);
            int pad_right = std::get<3>(num_padding_);
            
            if (padding_ == "same") 
            {
                int k_h = std::get<0>(kernel_size_);
                int k_w = std::get<1>(kernel_size_);
                int stride_h = std::get<0>(stride_);
                int stride_w = std::get<1>(stride_);

                // compute padding to maintain output size
                int output_h = (H_ + stride_h - 1) / stride_h; 
                int output_w = (W_ + stride_w - 1) / stride_w; 
                int pad_h_total = std::max(0, (output_h - 1) * stride_h + k_h - H_);
                int pad_w_total = std::max(0, (output_w - 1) * stride_w + k_w - W_);

                pad_top = pad_h_total / 2;
                pad_bottom = pad_h_total - pad_top;
                pad_left = pad_w_total / 2;
                pad_right = pad_w_total - pad_left;
              
            }
            else if (padding_ == "valid" || padding_ == "none") 
            {
                pad_top = 0;
                pad_bottom = 0;
                pad_left = 0;
                pad_right = 0;
            }

            int height = input.dimension(2);
            int width = input.dimension(3);
            
            if (padding_mode_ == "reflect") 
            {
                if (pad_top >= height || pad_bottom >= height || pad_left >= width || pad_right >= width) 
                {
                    throw std::runtime_error("Reflect padding size cannot be >= tensor dimension in layer: " + layer_name_);
                }
            }
          
            // Apply padding
            if (padding_mode_ == "zero")
            {
                // Zero padding using Eigen's pad function
                Eigen::array<std::pair<int, int>, 4> paddings = {
                    std::make_pair(0, 0),                        // No padding for batch dimension
                    std::make_pair(0, 0),                        // No padding for channel dimension
                    std::make_pair(pad_top, pad_bottom),         // Height padding
                    std::make_pair(pad_left, pad_right)          // Width padding
                };
                return input.pad(paddings);  
            }
            else if (padding_mode_ == "reflect")
            {
                return apply_manual_padding(input, pad_top, pad_bottom, pad_left, pad_right, "reflect");
            }
            else if (padding_mode_ == "edge")
            {
                return apply_manual_padding(input, pad_top, pad_bottom, pad_left, pad_right, "edge");
            }
            else
            {
                throw std::runtime_error("Unsupported padding mode: " + padding_mode_ + " in layer: " + layer_name_);
            }
        }

        Eigen::Tensor<double, 4> Conv2d::apply_manual_padding(
            const Eigen::Tensor<double, 4>& input,
            int pad_top, int pad_bottom, int pad_left, int pad_right,
            std::string padding_type)
        {
            int batch_size = input.dimension(0);
            int channels = input.dimension(1);
            int height = input.dimension(2);
            int width = input.dimension(3);
            
            Eigen::Tensor<double, 4> padded_input(
                batch_size, channels, 
                height + pad_top + pad_bottom, 
                width + pad_left + pad_right
            );
            padded_input.setZero();

            // Copy original input to the center
            padded_input.slice(
                Eigen::array<int, 4>{0, 0, pad_top, pad_left}, 
                Eigen::array<int, 4>{batch_size, channels, height, width}
            ) = input;

            // Determine if parallelization is beneficial
            const int total_elements = batch_size * channels * (height + pad_top + pad_bottom);
            const bool use_parallel = total_elements > 10000; // Configurable threshold

            // Apply vertical padding (top and bottom)
            apply_vertical_padding(padded_input, batch_size, channels, height, pad_top, pad_bottom, padding_type, use_parallel);
            
            // Apply horizontal padding (left and right)  
            apply_horizontal_padding(padded_input, batch_size, channels, height + pad_top + pad_bottom, width, pad_left, pad_right, padding_type, use_parallel);

            return padded_input;
        }

        void Conv2d::apply_vertical_padding(
            Eigen::Tensor<double, 4>& padded_input,
            int batch_size, int channels, int height,
            int pad_top, int pad_bottom, 
            std::string padding_type, bool use_parallel)
        {
            auto loop_body = [&](int b, int c) 
            {
                // Top padding
                for (int p = 0; p < pad_top; ++p) 
                {
                    int source_row;
                    if (padding_type == "reflect") 
                    {
                        source_row = pad_top + (pad_top - p - 1); // Mirror reflection
                    } 
                    else 
                    {
                        source_row = pad_top; // Extend edge value
                    }
                    padded_input.chip(p, 2).chip(c, 1).chip(b, 0) = padded_input.chip(source_row, 2).chip(c, 1).chip(b, 0);
                }
                
                // Bottom padding
                for (int p = 0; p < pad_bottom; ++p) 
                {
                    int source_row;
                    if (padding_type == "reflect") 
                    {
                        source_row = height + pad_top - p - 1; // Mirror reflection
                    } 
                    else 
                    { 
                        source_row = height + pad_top - 1; // Extend edge value
                    }
                    padded_input.chip(height + pad_top + p, 2).chip(c, 1).chip(b, 0) = padded_input.chip(source_row, 2).chip(c, 1).chip(b, 0);
                }
            };

            if (use_parallel) 
            {
                #pragma omp parallel for collapse(2)
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int c = 0; c < channels; ++c) 
                    {
                        loop_body(b, c);
                    }
                }
            } 
            else 
            {
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int c = 0; c < channels; ++c) 
                    {
                        loop_body(b, c);
                    }
                }
            }
        }

        void Conv2d::apply_horizontal_padding(
            Eigen::Tensor<double, 4>& padded_input,
            int batch_size, int channels, int padded_height, int width,
            int pad_left, int pad_right,
            std::string padding_type, bool use_parallel)
        {
            auto loop_body = [&](int b, int c, int h) 
            {
                // Left padding
                for (int p = 0; p < pad_left; ++p) 
                {
                    int source_col;
                    if (padding_type == "reflect") 
                    {
                        source_col = pad_left + (pad_left - p - 1); // Mirror reflection
                    } 
                    else 
                    {
                        source_col = pad_left; // Extend edge value
                    }
                    padded_input(b, c, h, p) = padded_input(b, c, h, source_col);
                }
                
                // Right padding
                for (int p = 0; p < pad_right; ++p) 
                {
                    int source_col;
                    if (padding_type == "reflect") 
                    {
                        source_col = width + pad_left - p - 1; // Mirror reflection
                    } 
                    else 
                    { 
                        source_col = width + pad_left - 1; // Extend edge value
                    }
                    padded_input(b, c, h, width + pad_left + p) = padded_input(b, c, h, source_col);
                }
            };

            if (use_parallel) 
            {
                #pragma omp parallel for collapse(3)
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int c = 0; c < channels; ++c) 
                    {
                        for (int h = 0; h < padded_height; ++h) 
                        {
                            loop_body(b, c, h);
                        }
                    }
                }
            } 
            else 
            {
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int c = 0; c < channels; ++c) 
                    {
                        for (int h = 0; h < padded_height; ++h) 
                        {
                            loop_body(b, c, h);
                        }
                    }
                }
            }
        }

        void Conv2d::init_output()
        {
            int k_h = std::get<0>(kernel_size_);
            int k_w = std::get<1>(kernel_size_);
            int stride_h = std::get<0>(stride_);
            int stride_w = std::get<1>(stride_);
            int pad_h = std::get<2>(num_padding_) + std::get<3>(num_padding_);
            int pad_w = std::get<0>(num_padding_) + std::get<1>(num_padding_);

            if (padding_ == "valid" || padding_ == "none") 
            {
                h_ = (H_ + pad_h - k_h) / stride_h + 1;
                w_ = (W_ + pad_w - k_w) / stride_w + 1;

                if ((H_ + pad_h - k_h) % stride_h != 0 || (W_ + pad_w - k_w) % stride_w != 0) 
                {
                    throw std::runtime_error("Non-integer output dimensions in layer: " + layer_name_);
                }
            } else if (padding_ == "same") 
            {
                h_ = std::ceil(static_cast<double>(H_) / stride_h);
                w_ = std::ceil(static_cast<double>(W_) / stride_w);
            }

            if (h_ <= 0 || w_ <= 0) 
            {
                throw std::runtime_error("Invalid output dimensions in layer: " + layer_name_);
            }

            output_ = Eigen::Tensor<double, 4>(B_, out_channels_, h_, w_);
            output_.setZero(); // Initialize output tensor with zeros
        }

        // Convert input tensor to column matrix for convolution
        Eigen::Tensor<double, 2> Conv2d::im2col(const Eigen::Tensor<double, 4>& input)
        {
            const int k_h = std::get<0>(kernel_size_);
            const int k_w = std::get<1>(kernel_size_);
            const int stride_h = std::get<0>(stride_);
            const int stride_w = std::get<1>(stride_);
            const int input_h = input.dimension(2);
            const int input_w = input.dimension(3);
            int out_h = (input_h - k_h) / stride_h + 1;
            int out_w = (input_w - k_w) / stride_w + 1;
            
            if (out_h <= 0 || out_w <= 0)
            {
                throw std::runtime_error("Invalid output dimensions in im2col for layer: " + layer_name_);
            }
            
            int col_height = k_h * k_w * in_channels_;
            int col_width = B_ * out_h * out_w;
            Eigen::Tensor<double, 2> col_matrix(col_height, col_width);
            col_matrix.setZero();
            
            // Parallelize outer loops with OpenMP
            // Use collapse to combine loops for better load balancing
            #pragma omp parallel for collapse(3)
            for (int b = 0; b < B_; ++b)
            {
                for (int oh = 0; oh < out_h; ++oh)
                {
                    for (int ow = 0; ow < out_w; ++ow)
                    {
                        int col_idx = b * out_h * out_w + oh * out_w + ow;
                        
                        int row_idx = 0;
                        for (int ic = 0; ic < in_channels_; ++ic)
                        {
                            for (int kh = 0; kh < k_h; ++kh)
                            {
                                for (int kw = 0; kw < k_w; ++kw)
                                {
                                    int ih = oh * stride_h + kh;
                                    int iw = ow * stride_w + kw;
                                    if (ih < input_h && iw < input_w)
                                    {
                                        col_matrix(row_idx, col_idx) = input(b, ic, ih, iw);
                                    }
                                    row_idx++;
                                }
                            }
                        }
                    }
                }
            }
            
            return col_matrix;
        }

        // Convert column matrix back to input tensor shape, accumulating gradients
        void Conv2d::col2im_add(const Eigen::Tensor<double, 2>& grad_input_col, Eigen::Tensor<double, 4>& grad_input)
        {
            const int k_h = std::get<0>(kernel_size_);
            const int k_w = std::get<1>(kernel_size_);
            const int stride_h = std::get<0>(stride_);
            const int stride_w = std::get<1>(stride_);
            const int input_h = grad_input.dimension(2);
            const int input_w = grad_input.dimension(3);
            
            // Parallelize outer loops with OpenMP
            // Use collapse to combine loops for better load balancing
            #pragma omp parallel for collapse(3)
            for (int b = 0; b < B_; ++b)
            {
                for (int oh = 0; oh < h_; ++oh)
                {
                    for (int ow = 0; ow < w_; ++ow)
                    {
                        int col_idx = b * h_ * w_ + oh * w_ + ow;
                        
                        int row_idx = 0;
                        for (int ic = 0; ic < in_channels_; ++ic)
                        {
                            for (int kh = 0; kh < k_h; ++kh)
                            {
                                for (int kw = 0; kw < k_w; ++kw)
                                {
                                    int ih = oh * stride_h + kh;
                                    int iw = ow * stride_w + kw;
                                    if (ih < input_h && iw < input_w)
                                    {
                                        // Accumulate gradients
                                        grad_input(b, ic, ih, iw) += grad_input_col(row_idx, col_idx);
                                    }
                                    row_idx++;
                                }
                            }
                        }
                    }
                }
            }
        }

        // helper method to set number of threads
        void Conv2d::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }

        // Forward pass
        Eigen::Tensor<double, 4> Conv2d::forward(const Eigen::Tensor<double, 4>& input)
        {
            // cache input shape for backward pass
            in_cache_ = input;
            B_ = input.dimension(0);
            C_ = input.dimension(1);
            H_ = input.dimension(2);
            W_ = input.dimension(3);
            
            // validate input
            if (C_ != in_channels_)
            {
                throw std::runtime_error("Input channels mismatch in layer: " + layer_name_);
            }
            if (H_ < std::get<0>(kernel_size_) || W_ < std::get<1>(kernel_size_))
            {
                throw std::runtime_error("Input dimensions too small for kernel in layer: " + layer_name_);
            }
            
            // pad input if needed
            Eigen::Tensor<double, 4> input_to_use = pad_input(input);
            
            // initialize output dimensions
            init_output();
            
            // im2col transformation
            Eigen::Tensor<double, 2> col_matrix = im2col(input_to_use);
            
            // reshape weights to matrix: [out_channels, k_h * k_w * in_channels]
            const int k_h = std::get<0>(kernel_size_);
            const int k_w = std::get<1>(kernel_size_);
            Eigen::Tensor<double, 2> reshaped_weight = weights_.reshape(
                Eigen::array<int, 2>{out_channels_, k_h * k_w * in_channels_}
            );
            
            int in_shape = reshaped_weight.dimension(0);   // out_channels_
            int cols = reshaped_weight.dimension(1);       // k_h * k_w * in_channels_
            int out_shape = col_matrix.dimension(1);       // B_ * h_ * w_
            
            Eigen::Tensor<double, 2> result(in_shape, out_shape);
            result.setZero();
            
            // Manual matrix multiplication with OpenMP parallelization
            // result(i, k) = sum_j(reshaped_weight(i, j) * col_matrix(j, k))
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < in_shape; ++i)
            {
                for (int k = 0; k < out_shape; ++k)
                {
                    double sum = 0.0;
                    for (int j = 0; j < cols; ++j)
                    {
                        sum += reshaped_weight(i, j) * col_matrix(j, k);
                    }
                    result(i, k) = sum;
                }
            }
            
            // add bias if applicable
            if (bias_)
            {
                Eigen::Tensor<double, 2> bias_broadcasted = biases_.reshape(
                    Eigen::array<int, 2>{out_channels_, 1}
                ).broadcast(
                    Eigen::array<int, 2>{1, B_ * h_ * w_}
                );
                result += bias_broadcasted;
            }
            
            // reshape to 4D tensor
            Eigen::Tensor<double, 4> pre_activation = result.reshape(
                Eigen::array<int, 4>{B_, out_channels_, h_, w_}
            );
            
            // apply activation  
            // output_ = activation_-> forward(pre_activation);
            return output_;
        }

        // Backward pass
        Eigen::Tensor<double, 4> Conv2d::backward(const Eigen::Tensor<double, 4>& grad_output)
        {
            // validate gradient dimensions
            if (grad_output.dimension(0) != B_ || grad_output.dimension(1) != out_channels_ ||
                grad_output.dimension(2) != h_ || grad_output.dimension(3) != w_)
            {
                throw std::runtime_error("Gradient output dimensions mismatch in layer: " + layer_name_);
            }
            
            const int k_h = std::get<0>(kernel_size_);
            const int k_w = std::get<1>(kernel_size_);
            
            // initialize gradients
            Eigen::Tensor<double, 4> grad_input(in_cache_.dimensions());
            grad_input.setZero();
            
            if (trainable_)
            {
                grad_weights_.setZero();
                if (bias_)
                {
                    grad_biases_.setZero();
                }
            }
            
            // pad input for gradient computation
            Eigen::Tensor<double, 4> input_to_use = pad_input(in_cache_);
            
            // compute gradient through activation
            // Eigen::Tensor<double, 4> grad_pre_activation = activation_->backward(grad_output);
            
            // Reshape grad_pre_activation to 2D: [out_channels, B*h*w]
            Eigen::Tensor<double, 2> grad_output_2d = grad_output.reshape(
                Eigen::array<int, 2>{out_channels_, B_ * h_ * w_}
            );
            
            // compute weight gradients
            if (trainable_)
            {
                Eigen::Tensor<double, 2> col_matrix = im2col(input_to_use);
                
                // Manual matrix multiplication for weight gradients with OpenMP
                // weight_grad = grad_output_2d * col_matrix^T
                // grad_output_2d: [out_channels, B*h*w]
                // col_matrix: [k_h*k_w*in_channels, B*h*w]
                // Result: [out_channels, k_h*k_w*in_channels]
                
                int weight_grad_rows = grad_output_2d.dimension(0);  // out_channels
                int weight_grad_cols = col_matrix.dimension(0);      // k_h*k_w*in_channels
                int inner_dim = grad_output_2d.dimension(1);         // B*h*w
                
                Eigen::Tensor<double, 2> weight_grad(weight_grad_rows, weight_grad_cols);
                weight_grad.setZero();
                
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < weight_grad_rows; ++i)
                {
                    for (int j = 0; j < weight_grad_cols; ++j)
                    {
                        double sum = 0.0;
                        for (int k = 0; k < inner_dim; ++k)
                        {
                            // Matrix multiplication: C(i,j) = sum_k A(i,k) * B^T(j,k) = sum_k A(i,k) * B(j,k)
                            sum += grad_output_2d(i, k) * col_matrix(j, k);
                        }
                        weight_grad(i, j) = sum;
                    }
                }
                
                // reshape back to 4D tensor
                grad_weights_ = weight_grad.reshape(
                    Eigen::array<int, 4>{out_channels_, in_channels_, k_h, k_w}
                );
                
                if (bias_)
                {
                    Eigen::Tensor<double, 1> bias_grad = grad_output_2d.sum(Eigen::array<int, 1>{1});
                    grad_biases_ = bias_grad;
                }
            }
            
            // compute input gradients
            // We need weights^T for backward: [k_h*k_w*in_channels, out_channels]
            Eigen::Tensor<double, 2> reshaped_weight = weights_.reshape(
                Eigen::array<int, 2>{out_channels_, k_h * k_w * in_channels_}
            );
            
            // Manual matrix multiplication: grad_input_col = reshaped_weight^T * grad_output_2d with OpenMP
            // reshaped_weight^T: [k_h*k_w*in_channels, out_channels]
            // grad_output_2d: [out_channels, B*h*w]
            // Result: [k_h*k_w*in_channels, B*h*w]
            
            int grad_input_rows = k_h * k_w * in_channels_;  // reshaped_weight transposed rows
            int grad_input_cols = grad_output_2d.dimension(1);  // B*h*w
            int inner_dim = out_channels_;                       // out_channels
            
            Eigen::Tensor<double, 2> grad_input_col(grad_input_rows, grad_input_cols);
            grad_input_col.setZero();
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < grad_input_rows; ++i)
            {
                for (int j = 0; j < grad_input_cols; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < inner_dim; ++k)
                    {
                        // Matrix multiplication: C(i,j) = sum_k A^T(i,k) * B(k,j) = sum_k A(k,i) * B(k,j)
                        sum += reshaped_weight(k, i) * grad_output_2d(k, j);
                    }
                    grad_input_col(i, j) = sum;
                }
            }
            
            // convert back to 4D tensor using col2im
            col2im_add(grad_input_col, grad_input);
            
            return grad_input;
        }

        //************************* MaxPool2D **************************//
        MaxPool2D::MaxPool2D(
            std::tuple<int, int> kernel_size,
            std::tuple<int, int> stride,
            std::string layer_name,
            std::string padding,
            std::tuple<int, int, int, int> num_padding,
            std::string padding_mode,
            std::string device,
            int parallel_threshold
        ) : kernel_size_(kernel_size),
            stride_(stride),
            layer_name_(layer_name),
            padding_(padding),
            num_padding_(num_padding),
            padding_mode_(padding_mode),
            device_(device),
            parallel_threshold_(parallel_threshold)
        {
            // Input validation
            if (std::get<0>(kernel_size_) <= 0 || std::get<1>(kernel_size_) <= 0) {
                throw std::runtime_error("Kernel size must be positive in layer: " + layer_name_);
            }
            if (std::get<0>(stride_) <= 0 || std::get<1>(stride_) <= 0) {
                throw std::runtime_error("Stride must be positive in layer: " + layer_name_);
            }
            if (padding_ != "valid" && padding_ != "same" && padding_ != "none") 
            {
                throw std::runtime_error("Invalid padding: " + padding_ + " in layer: " + layer_name_);
            }
            if (padding_mode_ != "zero" && padding_mode_ != "reflect" && padding_mode_ != "edge") 
            {
                throw std::runtime_error("Invalid padding mode: " + padding_mode_ + " in layer: " + layer_name_);
            }
            if (std::get<0>(num_padding_) < 0 || std::get<1>(num_padding_) < 0 || 
                std::get<2>(num_padding_) < 0 || std::get<3>(num_padding_) < 0) {
                throw std::runtime_error("Padding values must be non-negative in layer: " + layer_name_);
            }
        }

        Eigen::Tensor<double, 4> MaxPool2D::pad_input(const Eigen::Tensor<double, 4>& input)
        {
            int pad_top = std::get<0>(num_padding_);
            int pad_bottom = std::get<1>(num_padding_);
            int pad_left = std::get<2>(num_padding_);
            int pad_right = std::get<3>(num_padding_);
            
            if (padding_ == "same") 
            {
                int k_h = std::get<0>(kernel_size_);
                int k_w = std::get<1>(kernel_size_);
                int stride_h = std::get<0>(stride_);
                int stride_w = std::get<1>(stride_);

                // compute padding to maintain output size
                int output_h = (H_ + stride_h - 1) / stride_h; 
                int output_w = (W_ + stride_w - 1) / stride_w; 
                int pad_h_total = std::max(0, (output_h - 1) * stride_h + k_h - H_);
                int pad_w_total = std::max(0, (output_w - 1) * stride_w + k_w - W_);

                pad_top = pad_h_total / 2;
                pad_bottom = pad_h_total - pad_top;
                pad_left = pad_w_total / 2;
                pad_right = pad_w_total - pad_left;
              
            }
            else if (padding_ == "valid" || padding_ == "none") 
            {
                pad_top = 0;
                pad_bottom = 0;
                pad_left = 0;
                pad_right = 0;
            }

            int height = input.dimension(2);
            int width = input.dimension(3);

            if (padding_mode_ == "reflect") 
            {
                if (pad_top >= height || pad_bottom >= height || pad_left >= width || pad_right >= width) 
                {
                    throw std::runtime_error("Reflect padding size cannot be >= tensor dimension in layer: " + layer_name_);
                }
            }
          
            // Apply padding
            if (padding_mode_ == "zero")
            {
                // Zero padding using Eigen's pad function
                Eigen::array<std::pair<int, int>, 4> paddings = {
                    std::make_pair(0, 0),                        // No padding for batch dimension
                    std::make_pair(0, 0),                        // No padding for channel dimension
                    std::make_pair(pad_top, pad_bottom),         // Height padding
                    std::make_pair(pad_left, pad_right)          // Width padding
                };
                return input.pad(paddings);  
            }
            else if (padding_mode_ == "reflect")
            {
                return apply_manual_padding(input, pad_top, pad_bottom, pad_left, pad_right, "reflect");
            }
            else if (padding_mode_ == "edge")
            {
                return apply_manual_padding(input, pad_top, pad_bottom, pad_left, pad_right, "edge");
            }
            else
            {
                throw std::runtime_error("Unsupported padding mode: " + padding_mode_ + " in layer: " + layer_name_);
            }
        }
        Eigen::Tensor<double, 4> MaxPool2D::apply_manual_padding(
            const Eigen::Tensor<double, 4>& input,
            int pad_top, int pad_bottom, int pad_left, int pad_right,
            std::string padding_type)
        {
            int batch_size = input.dimension(0);
            int channels = input.dimension(1);
            int height = input.dimension(2);
            int width = input.dimension(3);
            
            Eigen::Tensor<double, 4> padded_input(
                batch_size, channels, 
                height + pad_top + pad_bottom, 
                width + pad_left + pad_right
            );
            padded_input.setZero();

            // Copy original input to the center
            padded_input.slice(
                Eigen::array<int, 4>{0, 0, pad_top, pad_left}, 
                Eigen::array<int, 4>{batch_size, channels, height, width}
            ) = input;

            // Use parallel_threshold_ and consider total workload better
            const int total_elements = batch_size * channels * (height + pad_top + pad_bottom);
            const bool use_parallel = total_elements > parallel_threshold_; 

            // Apply vertical padding (top and bottom)
            apply_vertical_padding(padded_input, batch_size, channels, height, pad_top, pad_bottom, padding_type, use_parallel);
            
            // Apply horizontal padding (left and right)  
            apply_horizontal_padding(padded_input, batch_size, channels, height + pad_top + pad_bottom, width, pad_left, pad_right, padding_type, use_parallel);

            return padded_input;
        }

        void MaxPool2D::apply_vertical_padding(
            Eigen::Tensor<double, 4>& padded_input,
            int batch_size, int channels, int height,
            int pad_top, int pad_bottom, 
            std::string padding_type, bool use_parallel)
        {
            auto loop_body = [&](int b, int c) 
            {
                // Top padding
                for (int p = 0; p < pad_top; ++p) 
                {
                    int source_row;
                    if (padding_type == "reflect") 
                    {
                        source_row = pad_top + (pad_top - p - 1); // Mirror reflection
                    } 
                    else 
                    {
                        source_row = pad_top; // Extend edge value
                    }
                    padded_input.chip(p, 2).chip(c, 1).chip(b, 0) = padded_input.chip(source_row, 2).chip(c, 1).chip(b, 0);
                }
                
                // Bottom padding
                for (int p = 0; p < pad_bottom; ++p) 
                {
                    int source_row;
                    if (padding_type == "reflect") 
                    {
                        source_row = height + pad_top - p - 1; // Mirror reflection
                    } 
                    else 
                    { 
                        source_row = height + pad_top - 1; // Extend edge value
                    }
                    padded_input.chip(height + pad_top + p, 2).chip(c, 1).chip(b, 0) = padded_input.chip(source_row, 2).chip(c, 1).chip(b, 0);
                }
            };

            if (use_parallel) 
            {
                #pragma omp parallel for collapse(2)
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int c = 0; c < channels; ++c) 
                    {
                        loop_body(b, c);
                    }
                }
            } 
            else 
            {
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int c = 0; c < channels; ++c) 
                    {
                        loop_body(b, c);
                    }
                }
            }
        }

        void MaxPool2D::apply_horizontal_padding(
            Eigen::Tensor<double, 4>& padded_input,
            int batch_size, int channels, int padded_height, int width,
            int pad_left, int pad_right,
            std::string padding_type, bool use_parallel)
        {
            auto loop_body = [&](int b, int c, int h) 
            {
                // Left padding
                for (int p = 0; p < pad_left; ++p) 
                {
                    int source_col;
                    if (padding_type == "reflect") 
                    {
                        source_col = pad_left + (pad_left - p - 1); // Mirror reflection
                    } 
                    else 
                    {
                        source_col = pad_left; // Extend edge value
                    }
                    padded_input(b, c, h, p) = padded_input(b, c, h, source_col);
                }
                
                // Right padding
                for (int p = 0; p < pad_right; ++p) 
                {
                    int source_col;
                    if (padding_type == "reflect") 
                    {
                        source_col = width + pad_left - p - 1; // Mirror reflection
                    } 
                    else 
                    { 
                        source_col = width + pad_left - 1; // Extend edge value
                    }
                    padded_input(b, c, h, width + pad_left + p) = padded_input(b, c, h, source_col);
                }
            };

            if (use_parallel) 
            {
                #pragma omp parallel for collapse(3)
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int c = 0; c < channels; ++c) 
                    {
                        for (int h = 0; h < padded_height; ++h) 
                        {
                            loop_body(b, c, h);
                        }
                    }
                }
            } 
            else 
            {
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int c = 0; c < channels; ++c) 
                    {
                        for (int h = 0; h < padded_height; ++h) 
                        {
                            loop_body(b, c, h);
                        }
                    }
                }
            }
        }

        void MaxPool2D::init_output()
        {
            int k_h = std::get<0>(kernel_size_);
            int k_w = std::get<1>(kernel_size_);
            int stride_h = std::get<0>(stride_);
            int stride_w = std::get<1>(stride_);
            int pad_h = std::get<2>(num_padding_) + std::get<3>(num_padding_);
            int pad_w = std::get<0>(num_padding_) + std::get<1>(num_padding_);

            if (padding_ == "valid" || padding_ == "none") 
            {
                h_ = (H_ + pad_h - k_h) / stride_h + 1;
                w_ = (W_ + pad_w - k_w) / stride_w + 1;

                if ((H_ + pad_h - k_h) % stride_h != 0 || (W_ + pad_w - k_w) % stride_w != 0) 
                {
                    throw std::runtime_error("Non-integer output dimensions in layer: " + layer_name_);
                }
            } else if (padding_ == "same") 
            {
                h_ = std::ceil(static_cast<double>(H_) / stride_h);
                w_ = std::ceil(static_cast<double>(W_) / stride_w);
            }

            if (h_ <= 0 || w_ <= 0) 
            {
                throw std::runtime_error("Invalid output dimensions in layer: " + layer_name_);
            }

            output_ = Eigen::Tensor<double, 4>(B_, C_, h_, w_);
            output_.setZero(); // Initialize output tensor with zeros    
        }

        // helper method to set number of threads
        void MaxPool2D::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }

        Eigen::Tensor<double, 4> MaxPool2D::forward(const Eigen::Tensor<double, 4>& input) 
        {
            B_ = input.dimension(0);
            C_ = input.dimension(1);
            H_ = input.dimension(2);
            W_ = input.dimension(3);
            
            // Validate input
            if (H_ < std::get<0>(kernel_size_) || W_ < std::get<1>(kernel_size_)) {
                throw std::runtime_error("Input dimensions too small for kernel in layer: " + layer_name_);
            }
            
            // Pad input if needed
            Eigen::Tensor<double, 4> input_to_use = pad_input(input);
            in_cache_ = input_to_use; // Cache padded input
            
            // Initialize output dimensions
            init_output();
            
            // Get kernel and stride parameters
            int kh = std::get<0>(kernel_size_);
            int kw = std::get<1>(kernel_size_);
            int sh = std::get<0>(stride_);
            int sw = std::get<1>(stride_);

            const int total_operations = B_ * C_ * h_ * w_;

            if (total_operations > parallel_threshold_) 
            {
                // Apply max pooling with OpenMP parallelization for large workloads
                #pragma omp parallel for collapse(4)
                for (int b = 0; b < B_; ++b) 
                {
                    for (int c = 0; c < C_; ++c) 
                    {
                        for (int oh = 0; oh < h_; ++oh) 
                        {
                            for (int ow = 0; ow < w_; ++ow) 
                            {
                                // Calculate input region bounds
                                int h_start = oh * sh;
                                int w_start = ow * sw;
                                int h_end = std::min(h_start + kh, static_cast<int>(input_to_use.dimension(2)));
                                int w_end = std::min(w_start + kw, static_cast<int>(input_to_use.dimension(3)));
                                
                                // Find maximum in the kernel window
                                double max_val = -std::numeric_limits<double>::infinity();
                                for (int kh_idx = h_start; kh_idx < h_end; ++kh_idx) 
                                {
                                    for (int kw_idx = w_start; kw_idx < w_end; ++kw_idx) 
                                    {
                                        max_val = std::max(max_val, input_to_use(b, c, kh_idx, kw_idx));
                                    }
                                }
                                
                                output_(b, c, oh, ow) = max_val;
                            }
                        }
                    }
                }
            } 
            else 
            {
                // Serial execution for small workloads to avoid OpenMP overhead
                for (int b = 0; b < B_; ++b) 
                {
                    for (int c = 0; c < C_; ++c) 
                    {
                        for (int oh = 0; oh < h_; ++oh) 
                        {
                            for (int ow = 0; ow < w_; ++ow) 
                            {
                                // Calculate input region bounds
                                int h_start = oh * sh;
                                int w_start = ow * sw;
                                int h_end = std::min(h_start + kh, static_cast<int>(input_to_use.dimension(2)));
                                int w_end = std::min(w_start + kw, static_cast<int>(input_to_use.dimension(3)));
                                
                                // Find maximum in the kernel window
                                double max_val = -std::numeric_limits<double>::infinity();
                                for (int kh_idx = h_start; kh_idx < h_end; ++kh_idx) 
                                {
                                    for (int kw_idx = w_start; kw_idx < w_end; ++kw_idx) 
                                    {
                                        max_val = std::max(max_val, input_to_use(b, c, kh_idx, kw_idx));
                                    }
                                }
                                
                                output_(b, c, oh, ow) = max_val;
                            }
                        }
                    }
                }
            }
            
            return output_;
        }

        Eigen::Tensor<double, 4> MaxPool2D::backward(const Eigen::Tensor<double, 4>& grad_output) 
        {
            // Initialize gradient tensor with same dimensions as cached input
            Eigen::Tensor<double, 4> grad_input_padded = Eigen::Tensor<double, 4>(
                in_cache_.dimension(0), in_cache_.dimension(1), 
                in_cache_.dimension(2), in_cache_.dimension(3)
            );
            grad_input_padded.setZero();
            
            int kh = std::get<0>(kernel_size_);
            int kw = std::get<1>(kernel_size_);
            int sh = std::get<0>(stride_);
            int sw = std::get<1>(stride_);

            const int total_operations = B_ * C_ * h_ * w_;
            
            if (total_operations > parallel_threshold_) 
            {
                // Parallel execution with atomic operations to prevent race conditions
                #pragma omp parallel for collapse(4)
                for (int b = 0; b < B_; ++b) 
                {
                    for (int c = 0; c < C_; ++c) 
                    {
                        for (int oh = 0; oh < h_; ++oh) 
                        {
                            for (int ow = 0; ow < w_; ++ow) 
                            {
                                // Calculate input region bounds
                                int h_start = oh * sh;
                                int w_start = ow * sw;
                                int h_end = std::min(h_start + kh, static_cast<int>(in_cache_.dimension(2)));
                                int w_end = std::min(w_start + kw, static_cast<int>(in_cache_.dimension(3)));
                                
                                // Find the position of maximum value in the kernel window
                                double max_val = -std::numeric_limits<double>::infinity();
                                int max_h = h_start;
                                int max_w = w_start;
                                
                                for (int kh_idx = h_start; kh_idx < h_end; ++kh_idx) 
                                {
                                    for (int kw_idx = w_start; kw_idx < w_end; ++kw_idx) 
                                    {
                                        if (in_cache_(b, c, kh_idx, kw_idx) > max_val) 
                                        {
                                            max_val = in_cache_(b, c, kh_idx, kw_idx);
                                            max_h = kh_idx;
                                            max_w = kw_idx;
                                        }
                                    }
                                }
                                
                                // Use atomic operation to prevent race condition
                                #pragma omp atomic
                                grad_input_padded(b, c, max_h, max_w) += grad_output(b, c, oh, ow);
                            }
                        }
                    }
                }
            } 
            else 
            {
                // Serial execution for small workloads
                for (int b = 0; b < B_; ++b) 
                {
                    for (int c = 0; c < C_; ++c) 
                    {
                        for (int oh = 0; oh < h_; ++oh) 
                        {
                            for (int ow = 0; ow < w_; ++ow) 
                            {
                                // Calculate input region bounds
                                int h_start = oh * sh;
                                int w_start = ow * sw;
                                int h_end = std::min(h_start + kh, static_cast<int>(in_cache_.dimension(2)));
                                int w_end = std::min(w_start + kw, static_cast<int>(in_cache_.dimension(3)));
                                
                                // Find the position of maximum value in the kernel window
                                double max_val = -std::numeric_limits<double>::infinity();
                                int max_h = h_start;
                                int max_w = w_start;
                                
                                for (int kh_idx = h_start; kh_idx < h_end; ++kh_idx) 
                                {
                                    for (int kw_idx = w_start; kw_idx < w_end; ++kw_idx) 
                                    {
                                        if (in_cache_(b, c, kh_idx, kw_idx) > max_val) 
                                        {
                                            max_val = in_cache_(b, c, kh_idx, kw_idx);
                                            max_h = kh_idx;
                                            max_w = kw_idx;
                                        }
                                    }
                                }
                                
                                grad_input_padded(b, c, max_h, max_w) += grad_output(b, c, oh, ow);
                            }
                        }
                    }
                }
            }
            
            // Check if any padding was actually applied
            bool padding_applied = false;
            int pad_top = 0, pad_left = 0;
            
            if (padding_ == "same") 
            {
                // Calculate padding for "same" mode
                int k_h = std::get<0>(kernel_size_);
                int k_w = std::get<1>(kernel_size_);
                int stride_h = std::get<0>(stride_);
                int stride_w = std::get<1>(stride_);
                
                int output_h = (H_ + stride_h - 1) / stride_h;
                int output_w = (W_ + stride_w - 1) / stride_w;
                int pad_h_total = std::max(0, (output_h - 1) * stride_h + k_h - H_);
                int pad_w_total = std::max(0, (output_w - 1) * stride_w + k_w - W_);
                
                pad_top = pad_h_total / 2;
                pad_left = pad_w_total / 2;
                padding_applied = (pad_h_total > 0 || pad_w_total > 0);
            }
            else if (padding_ != "valid" && padding_ != "none")
            {
                pad_top = std::get<0>(num_padding_);
                pad_left = std::get<2>(num_padding_);
                padding_applied = (pad_top > 0 || std::get<1>(num_padding_) > 0 || 
                                 pad_left > 0 || std::get<3>(num_padding_) > 0);
            }
            
            // Return padded gradient if no padding was applied
            if (!padding_applied) 
            {
                return grad_input_padded;
            }
            
            // Extract the original input region from padded gradient
            Eigen::Tensor<double, 4> grad_input(B_, C_, H_, W_);
            
            if (total_operations > parallel_threshold_) 
            {
                #pragma omp parallel for collapse(4)
                for (int b = 0; b < B_; ++b) 
                {
                    for (int c = 0; c < C_; ++c) 
                    {
                        for (int h = 0; h < H_; ++h) 
                        {
                            for (int w = 0; w < W_; ++w) 
                            {
                                grad_input(b, c, h, w) = grad_input_padded(b, c, h + pad_top, w + pad_left);
                            }
                        }
                    }
                }
            } 
            else 
            {
                // Serial execution for small workloads
                for (int b = 0; b < B_; ++b) 
                {
                    for (int c = 0; c < C_; ++c) 
                    {
                        for (int h = 0; h < H_; ++h) 
                        {
                            for (int w = 0; w < W_; ++w) 
                            {
                                grad_input(b, c, h, w) = grad_input_padded(b, c, h + pad_top, w + pad_left);
                            }
                        }
                    }
                }
            }
            
            return grad_input;
        }

        /*************************************** Flatten ********************************************/
        Flatten::Flatten(std::string layer_name) : layer_name_(layer_name) {}

        Eigen::Tensor<double, 2> Flatten::forward(const Eigen::Tensor<double, 4>& input) 
        {
            if (input.size() == 0) 
            {
                throw std::runtime_error("Flatten: Empty input tensor");
            }

            // save input shape
            for (int i = 0; i < 4; ++i) 
            {
                in_shape_4d[i] = input.dimension(i);
            }

            B_ = input.dimension(0);
            C_ = input.dimension(1);
            H_ = input.dimension(2);
            W_ = input.dimension(3);

            grad_input_ = Eigen::Tensor<double, 4>(input.dimensions());
            grad_input_.setZero();

            // compute flattened shape
            Eigen::Index flat_dim1 = C_ * H_ * W_;
            Eigen::Index flat_dim0 = B_;

            std::cout << "flat_dim0: " << flat_dim0 << ", flat_dim1: " << flat_dim1 << std::endl;

            // reshape and copy to new tensor
            Eigen::array<Eigen::Index, 2> new_shape = {flat_dim0, flat_dim1};
            Eigen::Tensor<double, 2> flattened_input = input.reshape(new_shape);
            

            return flattened_input;
        }

        Eigen::Tensor<double, 4> Flatten::backward(const Eigen::Tensor<double, 2>& grad_output) 
        {
            // Manual copying to ensure correct data layout
            // Convert from [B, C*H*W] back to [B, C, H, W]
            Eigen::Index expected_features = C_ * H_ * W_;
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < B_; ++b)
            {
                for (Eigen::Index f = 0; f < expected_features; ++f)
                {
                    // Convert flat index back to 3D coordinates
                    int c = f / (H_ * W_);
                    int remaining = f % (H_ * W_);
                    int h = remaining / W_;
                    int w = remaining % W_;
                    
                    grad_input_(b, c, h, w) = grad_output(b, f);
                }
            }
            return grad_input_;
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