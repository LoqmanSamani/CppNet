#ifndef LINEAR_HPP
#define LINEAR_HPP

#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>
#include "CppNet/layers/layer.hpp"


// header file for Linear (Fully Connected: Dense) Layer implementation

namespace CppNet
{
    namespace Layers
    {
        class Linear : public Layer
        {
            public:
                Linear(
                    int in_size, 
                    int out_size, 
                    std::string layer_name = "Linear", 
                    bool trainable = true, 
                    bool bias = true,
                    std::string device = "gpu",
                    std::string weight_init = "xavier"
                ); 
                ~Linear();
                
                const Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input);
                const Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output);

                void reset_grads() 
                {
                    if (trainable_) 
                    {
                        grad_weights_.setZero();
                        if (bias_ && grad_biases_.size() > 0) 
                            grad_biases_.setZero();    
                    } 
                }

                int get_input_size() const { return in_size_; }
                int get_output_size() const { return out_size_; }
                std::string get_layer_name() const { return layer_name_; }
                std::string get_device() { return device_; }

                Eigen::Tensor<float, 2>& get_weights() { return weights_; }
                const Eigen::Tensor<float, 2>& get_weights() const { return weights_; }
                Eigen::Tensor<float, 1>& get_biases() { return biases_; }
                const Eigen::Tensor<float, 1>& get_biases() const { return biases_; }

                const Eigen::Tensor<float, 2>& get_grad_weights() const { return grad_weights_; }
                const Eigen::Tensor<float, 1>& get_grad_biases() const { return grad_biases_; }

                void set_weights(const Eigen::Tensor<float, 2>& weights) { weights_ = weights; }
                void set_biases(const Eigen::Tensor<float, 1>& biases) { biases_ = biases; }

                bool is_trainable() const override { return trainable_; }

                void freeze(){ trainable_ = false; }
                void unfreeze() { trainable_ = true; }

                bool has_bias() const { return bias_; }

                void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

                static void set_num_threads(int num_threads);
                void set_max_batch_size(int max_batch_size);
                
                void print_layer_info() const 
                {
                    std::cout << "  Layer: " << layer_name_ << std::endl;
                    std::cout << "  Input size: " << in_size_ << std::endl;
                    std::cout << "  Output size: " << out_size_ << std::endl;
                    std::cout << "  Trainable: " << (trainable_ ? "Yes" : "No") << std::endl;
                    std::cout << "  Has bias: " << (bias_ ? "Yes" : "No") << std::endl;
                    std::cout << "  Weight shape: [" << weights_.dimension(0) << ", " << weights_.dimension(1) << "]" << std::endl;

                    if (bias_) 
                    {
                        std::cout << "  Bias shape: [" << biases_.dimension(0) << "]" << std::endl;
                    }
                }

                #ifdef USE_CUDA
                    void init_gpu_buffers(int max_batch_size);
                    void cleanup_gpu_buffers();
                    void sync_weights_to_gpu();
                    void sync_weights_from_gpu();
                    void sync_gradients_from_gpu();
                    
                    float* get_d_weights() { return d_weights_; }
                    float* get_d_bias() { return d_bias_; }
                    float* get_d_grad_weights() { return d_grad_weights_; }
                    float* get_d_grad_bias() { return d_grad_bias_; }
                    
                    bool is_gpu_initialized() const { return gpu_initialized_; }
                #endif

            private:
                int in_size_;
                int out_size_;
                std::string layer_name_;
                bool trainable_;
                bool bias_;
                std::string device_;
                std::string weight_init_;   
                Eigen::Tensor<float, 2> weights_;
                Eigen::Tensor<float, 1> biases_;
                Eigen::Tensor<float, 2> in_cache_;
                Eigen::Tensor<float, 2> grad_weights_;
                Eigen::Tensor<float, 1> grad_biases_;

                // GPU output caches (CPU-side storage for return values)
                Eigen::Tensor<float, 2> gpu_output_cache_;
                Eigen::Tensor<float, 2> gpu_grad_input_cache_;

                void reinitialize_weights(const std::string& new_init_method);
                void init_params_and_grads();
                void forward_cpu(
                    const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_, 
                    const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output, 
                    int batch_size, int input_size, int output_size, bool bias_);

                void forward_gpu(
                    const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_, 
                    const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output, 
                    int batch_size, int input_size, int output_size, bool bias_);

                void forward_eigen(
                    const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_, 
                    const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output, 
                    int batch_size, int input_size, int output_size, bool bias_);

                void backward_gpu(int batch_size, int output_size, int input_size);
                
                void backward_cpu(
                    const Eigen::Tensor<float, 2>& grad_output, const Eigen::Tensor<float, 2>& in_cache_,
                    const Eigen::Tensor<float, 2>& weights_, Eigen::Tensor<float, 2>& grad_weights_, 
                    Eigen::Tensor<float, 1>& grad_biases_, Eigen::Tensor<float, 2>& grad_input,  
                    int batch_size, int output_size, int input_size, bool trainable_, bool bias_);

                void backward_eigen(
                    const Eigen::Tensor<float, 2>& grad_output, const Eigen::Tensor<float, 2>& in_cache_,
                    const Eigen::Tensor<float, 2>& weights_, Eigen::Tensor<float, 2>& grad_weights_, 
                    Eigen::Tensor<float, 1>& grad_biases_, Eigen::Tensor<float, 2>& grad_input,  
                    int batch_size, int output_size, int input_size, bool trainable_, bool bias_);
                
                #ifdef USE_CUDA
                    float* d_weights_;      
                    float* d_bias_;          
                    float* d_input_cache_;
                    float* d_output_;         
                    float* d_grad_weights_;   
                    float* d_grad_bias_;      
                    float* d_grad_input_;
                    float* d_grad_output_;
                    
                    bool gpu_initialized_;    
                    int gpu_max_batch_size_;
                #endif                  
        };
    }
}

#endif // LINEAR_HPP