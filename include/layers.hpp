#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <tuple>


namespace CppNet 
{
    class Optimizer; // forward declaration

    namespace Layers
    {
        

        class Layer {
            public:
                virtual bool is_trainable() const = 0;
                virtual void update_parameters(Optimizer& optimizer, double learning_rate) = 0;
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

                void update_parameters(Optimizer& optimizer, double learning_rate) override;
                
                void print_layer_info() const 
                {
                    std::cout << "Layer: " << layer_name_ << std::endl;
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
                bool bias_;
                std::string layer_name_;
                Eigen::Tensor<double, 2> weights_;
                Eigen::Tensor<double, 1> biases_;
                Eigen::Tensor<double, 2> in_cache_;
                Eigen::Tensor<double, 2> grad_weights_;
                Eigen::Tensor<double, 1> grad_biases_;
                bool trainable_; // if gradient should be calculated. if false: layer is frozen.

                void init_params_and_grads();
        };

    }
    
}

#endif // LAYERS_HPP