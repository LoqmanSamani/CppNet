#include <Eigen/Dense>
#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"





namespace CppNet
{
    namespace Layers
    {
        // a simple implementation of a linear(dense) layer  
        Linear::Linear(int in_size, int out_size, std::string layer_name, bool trainable, bool bias) 
            : trainable_(trainable), in_size_(in_size), out_size_(out_size), bias_(bias), layer_name_(layer_name)
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

            // scaling factor for Xavier initialization
            double scale = std::sqrt(6.0 / (in_size_ + out_size_));
            std::uniform_real_distribution<> dis(-scale, scale);

            // initialize weight-tensor
            weights_ = Eigen::Tensor<double, 2>(in_size_, out_size_);

            for (int i = 0; i < in_size_; ++i) 
            {
                for (int j = 0; j < out_size_; ++j) 
                {
                    weights_(i, j) = dis(gen);
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

        void Linear::update_parameters(Optimizer& optimizer, double learning_rate) {
            optimizer.update(*this, learning_rate);
        }
    
        Eigen::Tensor<double, 2> Linear::forward(const Eigen::Tensor<double, 2>& X)
        {
            // check dimensions
            if (X.dimension(1) != weights_.dimension(0))
            {
                throw std::runtime_error("Shape mismatch: in layer: " + layer_name_ + " X.dimension(1) must be equal weights_.dimension(0)!");
            }
            // store X to use later in gradient calculation
            in_cache_ = X;
            
            // forward calculation
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> output = X.contract(weights_, product_dims);

            if (bias_) {
                // add bias to each row - broadcasting the bias vector
                Eigen::array<int, 2> broadcast_dims({X.dimension(0), 1});
                Eigen::Tensor<double, 2> bias_broadcasted = biases_.reshape(Eigen::array<int, 2>({1, biases_.dimension(0)})).broadcast(broadcast_dims);
                output = output + bias_broadcasted;
            }
            
            return output;
        }

        Eigen::Tensor<double, 2> Linear::backward(const Eigen::Tensor<double, 2>& grad_out)
        {
            // dimension validation
            if (grad_out.dimension(0) != in_cache_.dimension(0)) 
            {
                throw std::runtime_error("Batch size mismatch in layer: " + layer_name_);
            }
            if (grad_out.dimension(1) != out_size_) 
            {
                throw std::runtime_error("Output size mismatch in layer: " + layer_name_);
            }
            
            // compute parameter gradients (only if trainable)
            if (trainable_) {
                // gradient w.r.t. weights: X^T * grad_out
                Eigen::array<int, 2> transpose_dims({1, 0});
                Eigen::Tensor<double, 2> X_transposed = in_cache_.shuffle(transpose_dims);
                
                Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
                grad_weights_ = X_transposed.contract(grad_out, product_dims);
                
                // gradient w.r.t. biases: sum over batch dimension
                if (bias_) 
                {
                    Eigen::array<int, 1> batch_dim({0});
                    grad_biases_ = grad_out.sum(batch_dim);
                }
            }
            
            // compute gradient w.r.t. input: grad_out * W^T
            Eigen::array<int, 2> transpose_dims({1, 0});
            Eigen::Tensor<double, 2> weights_transposed = weights_.shuffle(transpose_dims);
            
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> grad_input = grad_out.contract(weights_transposed, product_dims);
            
            return grad_input;
        }

    }
    

}