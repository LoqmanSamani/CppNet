#ifndef LINEAR_HPP
#define LINEAR_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>



namespace CppNet
{
    namespace Optimizers
    {
        class Optimizer;
    }
    
    namespace Layers
    {
        
        class Layer 
        {
            public:
                virtual bool is_trainable() const = 0;
                virtual void step(Optimizers::Optimizer& optimizer, float learning_rate) = 0;
                virtual ~Layer() = default;
        };

        //********************* Linear (Fully Connected: Dense) Layer *********************//
        class Linear : public Layer
        {
            public:

                Linear(
                    int in_size, 
                    int out_size, 
                    std::string layer_name = "Linear", 
                    bool trainable = true, 
                    bool bias = true,
                    std::string device = "cpu",
                    std::string weight_init = "xavier",
                    int parallel_threshold = 10000
                ); 
                
                Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input);
                
                void Linear::forward_cpu(
                    const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_, 
                    const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output, 
                    int batch_size, int input_size, int output_size);

                void Linear::forward_gpu(
                    const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_, 
                    const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output, 
                    int batch_size, int input_size, int output_size);

                void Linear::forward_eigen(
                    const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_, 
                    const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output, 
                    int batch_size, int input_size, int output_size);

                Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output);

                void Linear::backward_gpu(
                    const Eigen::Tensor<float, 2>& grad_output, const Eigen::Tensor<float, 2>& in_cache_,
                    const Eigen::Tensor<float, 2>& weights_, Eigen::Tensor<float, 2>& grad_weights_, 
                    Eigen::Tensor<float, 1>& grad_biases_, Eigen::Tensor<float, 2>& grad_input,  
                    int batch_size, int output_size, int input_size, bool trainable_, bool bias_);
                
                void Linear::backward_cpu(
                    const Eigen::Tensor<float, 2>& grad_output, const Eigen::Tensor<float, 2>& in_cache_,
                    const Eigen::Tensor<float, 2>& weights_, Eigen::Tensor<float, 2>& grad_weights_, 
                    Eigen::Tensor<float, 1>& grad_biases_, Eigen::Tensor<float, 2>& grad_input,  
                    int batch_size, int output_size, int input_size, bool trainable_, bool bias_);

                void Linear::backward_eigen(
                    const Eigen::Tensor<float, 2>& grad_output, const Eigen::Tensor<float, 2>& in_cache_,
                    const Eigen::Tensor<float, 2>& weights_, Eigen::Tensor<float, 2>& grad_weights_, 
                    Eigen::Tensor<float, 1>& grad_biases_, Eigen::Tensor<float, 2>& grad_input,  
                    int batch_size, int output_size, int input_size, bool trainable_, bool bias_);

                void reset_grads() 
                    {

                        if (trainable_) 
                        {
                            grad_weights_.setZero();
                            if (bias_ && grad_biases_.size() > 0) 
                            { 
                                grad_biases_.setZero();
                            }
                        } 
                    }

                int get_input_size() const { return in_size_; }
                int get_output_size() const { return out_size_; }
                std::string get_layer_name() const { return layer_name_; }

                Eigen::Tensor<float, 2>& get_weights() { return weights_; }
                const Eigen::Tensor<float, 2>& get_weights() const { return weights_; }
                Eigen::Tensor<float, 1>& get_biases() { return biases_; }
                const Eigen::Tensor<float, 1>& get_biases() const { return biases_; }

                const Eigen::Tensor<float, 2>& get_grad_weights() const { return grad_weights_; }
                const Eigen::Tensor<float, 1>& get_grad_biases() const { return grad_biases_; }

                void set_weights(const Eigen::Tensor<float, 2>& weights) { weights_ = weights; }
                void set_biases(const Eigen::Tensor<float, 1>& biases) { biases_ = biases; }

                bool is_trainable() const override { return trainable_; }

                void freeze(){ trainable_ = false; } // freeze the parameters of the layer.
                void unfreeze() { trainable_ = true; } // unfreeze the parameters of the layer.

                bool has_bias() const { return bias_; }

                void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

                // OpenMP utility method
                static void set_num_threads(int num_threads);

                
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

            private:
          
                int in_size_;
                int out_size_;
                std::string layer_name_;
                bool trainable_; // if gradient should be calculated. if false: layer is frozen.
                bool bias_;
                std::string device_;
                std::string weight_init_;
                int parallel_threshold_; // threshold for input size to enable parallelization    
                Eigen::Tensor<float, 2> weights_;
                Eigen::Tensor<float, 1> biases_;
                Eigen::Tensor<float, 2> in_cache_;
                Eigen::Tensor<float, 2> grad_weights_;
                Eigen::Tensor<float, 1> grad_biases_;

                void reinitialize_weights(const std::string& new_init_method);
                void init_params_and_grads();
                         
        };
    }
}


#endif // LINEAR_HPP