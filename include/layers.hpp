#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <iostream>
//#include "activations.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <tuple>


namespace CppNet 
{

    namespace Optimizers 
    {
        class Optimizer; // forward declaration
    }
    namespace Activations {
        class Activation;
        class ReLU;
    }

    namespace Layers
    {
         
        class Layer {
            public:
                virtual bool is_trainable() const = 0;
                virtual void update_parameters(Optimizers::Optimizer& optimizer, double learning_rate) = 0;
                virtual ~Layer() = default;
        };

        class Linear : public Layer
        {
            public:
                
                Linear(
                    int in_size, 
                    int out_size, 
                    std::string layer_name = "Linear", 
                    bool trainable = true, 
                    bool bias = true
                );

                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& X);

                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_out);

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

                Eigen::Tensor<double, 2>& get_weights() { return weights_; }
                const Eigen::Tensor<double, 2>& get_weights() const { return weights_; }
                Eigen::Tensor<double, 1>& get_biases() { return biases_; }
                const Eigen::Tensor<double, 1>& get_biases() const { return biases_; }

                const Eigen::Tensor<double, 2>& get_grad_weights() const { return grad_weights_; }
                const Eigen::Tensor<double, 1>& get_grad_biases() const { return grad_biases_; }

                void set_weights(const Eigen::Tensor<double, 2>& weights) { weights_ = weights; }
                void set_biases(const Eigen::Tensor<double, 1>& biases) { biases_ = biases; }

                bool is_trainable() const override { return trainable_; }

                void freeze(){ trainable_ = false; } // freeze the parameters of the layer.
                void unfreeze() { trainable_ = true; } // unfreeze the parameters of the layer.

                bool has_bias() const { return bias_; }

                void update_parameters(Optimizers::Optimizer& optimizer, double learning_rate) override;
                
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
                bool trainable_; // if gradient should be calculated. if false: layer is frozen.
                int in_size_;
                int out_size_;
                bool bias_;
                std::string layer_name_;
                Eigen::Tensor<double, 2> weights_;
                Eigen::Tensor<double, 1> biases_;
                Eigen::Tensor<double, 2> in_cache_;
                Eigen::Tensor<double, 2> grad_weights_;
                Eigen::Tensor<double, 1> grad_biases_;
                

                void init_params_and_grads();
        };

        class Conv2d: public Layer
        {
            public:

                Conv2d(
                    int in_channels,
                    int out_channels,
                    Activations::Activation* activator = nullptr,
                    std::tuple<int, int> kernel_size = std::make_tuple(3, 3),
                    std::tuple<int, int> stride = std::make_tuple(1, 1),
                    std::string padding = "valid",
                    std::tuple<int, int, int, int> num_padding = std::make_tuple(1, 1, 1, 1),
                    std::string padding_mode = "zero",
                    std::string layer_name = "Conv2D",
                    bool trainable = true,
                    bool bias = true
                );

                Eigen::Tensor<double, 4> forward(Eigen::Tensor<double, 4>& X);
                Eigen::Tensor<double, 4> backward(Eigen::Tensor<double, 4>& grad_out);

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


                int get_input_channels() const { return in_channels_; }
                int get_output_channels() const { return out_channels_; }
                std::string get_layer_name() const { return layer_name_; }

                Eigen::Tensor<double, 4>& get_weights() { return weights_; }
                const Eigen::Tensor<double, 4>& get_weights() const { return weights_; }
                Eigen::Tensor<double, 1>& get_biases() { return biases_; }
                const Eigen::Tensor<double, 1>& get_biases() const { return biases_; }

                const Eigen::Tensor<double, 4>& get_grad_weights() const { return grad_weights_; }
                const Eigen::Tensor<double, 1>& get_grad_biases() const { return grad_biases_; }

                void set_weights(const Eigen::Tensor<double, 2>& weights) { weights_ = weights; }
                void set_biases(const Eigen::Tensor<double, 1>& biases) { biases_ = biases; }

                bool is_trainable() const override { return trainable_; }
                void freeze(){ trainable_ = false; } // freeze the parameters of the layer.
                void unfreeze() { trainable_ = true; } // unfreeze the parameters of the layer.
                bool has_bias() const { return bias_; }

                void update_parameters(Optimizers::Optimizer& optimizer, double learning_rate) override;
                
                void print_layer_info() const 
                {
                    std::cout << "Layer: " << layer_name_ << std::endl;
                    std::cout << "  Input channels: " << in_channels_ << std::endl;
                    std::cout << "  Output channels: " << out_channels_ << std::endl;
                    std::cout << "  Trainable: " << (trainable_ ? "Yes" : "No") << std::endl;
                    std::cout << "  Has bias: " << (bias_ ? "Yes" : "No") << std::endl;
                    std::cout << "  Weight shape: [" << weights_.dimension(0) << ", " << weights_.dimension(1) << "]" << std::endl;

                    if (bias_) 
                    {
                        std::cout << "  Bias shape: [" << biases_.dimension(0) << "]" << std::endl;
                    }
                }

            private:

                int in_channels_;
                int out_channels_;
                std::unique_ptr<Activations::ReLU> default_relu_;
                Activations::Activation* activator_;
                std::tuple<int, int> stride_;
                std::string padding_;
                std::tuple<int, int, int, int> num_padding_;
                std::string padding_mode_;
                std::string layer_name_;
                bool trainable_;
                bool bias_;
                std::tuple<int, int> kernel_size_;
                
                Eigen::Tensor<double, 4> weights_;
                Eigen::Tensor<double, 1> biases_;
                Eigen::Tensor<double, 4> grad_weights_;
                Eigen::Tensor<double, 1> grad_biases_;
                Eigen::Tensor<double, 4> in_cache_; 
                Eigen::Tensor<double, 4> output_;

                // input/output dimensions
                int B_, C_, H_, W_;  // input dimensions
                int h_, w_;          // output dimensions
                
                void init_params_and_grads();
                void init_output();
                Eigen::Tensor<double, 4> pad_input(const Eigen::Tensor<double, 4>& X);
        };
    
    }
}




#endif // LAYERS_HPP