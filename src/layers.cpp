#include <Eigen/Dense>
#include "layers.hpp"
#include "optimizers.hpp"





namespace CppNet
{
    // a simple implementation of a linear(dense) layer  
    Linear::Linear(int in_size, int out_size, std::string layer_name, bool trainable, bool bias) 
        : trainable_(trainable), in_size_(in_size), out_size_(out_size), bias_(bias), layer_name_(layer_name)
    {
        // check if in and out sizes are positive integers
        if (in_size <= 0 || out_size <= 0)
        {
            throw std::runtime_error("in_size and out_size must be positive integers!");
        }

        // initialize parameters and gradients
        init_params_and_grads();
    }

    void Linear::init_params_and_grads()
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        double scale = std::sqrt(6.0 / (in_size_ + out_size_));
        std::uniform_real_distribution<> dis(-scale, scale);

        // initializing weights with Xavier method (default method).
        // one can later change the initialized parameters!
        weights_ = Eigen::MatrixXd::NullaryExpr(in_size_, out_size_, [&]() { return dis(gen); });

        // initialize weight-gradient matrix with zero
        grad_weights_ = Eigen::MatrixXd::Zero(in_size_, out_size_);

        if (bias_)
        {
            // initialize biases with zero, if bias_ is true.
            biases_ = Eigen::VectorXd::Zero(out_size_); 
            // initialize bias-gradient matrix with zero 
            grad_biases_ = Eigen::VectorXd::Zero(out_size_);
        }
        else
        {
            // initialize empty biases and gradients when bias is false
            biases_ = Eigen::VectorXd(0);
            grad_biases_ = Eigen::VectorXd(0);
        }
    }

    void Linear::update_parameters(Optimizer& optimizer, double learning_rate) {
        optimizer.update(*this, learning_rate);
    }

    Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd& X)
    {
        // check dimensions
        if (X.cols() != weights_.rows())
        {
            throw std::runtime_error("Shape mismatch: X.cols() must be equal weights_.rows()!");
        }
        // store X to use later in gradient calculation
        in_cache_ = X;

        Eigen::MatrixXd output = X * weights_;
        if (bias_)
        {
            output.rowwise() += biases_.transpose();
        }
        
        return output;
    }

    Eigen::MatrixXd Linear::backward(const Eigen::MatrixXd& grad_out)
    {
        // check dimensions.
        if (grad_out.rows() != in_cache_.rows())
        {
            throw std::runtime_error("Shape mismatch: grad_out.rows() must be equal in_cache_.rows()!");
        }
        if (grad_out.cols() != out_size_)
        {
            throw std::runtime_error("Shape mismatch: grad_out.cols() must be equal out_size_!");
        }
        
        // check if layer is not frozen
        if (trainable_)
        {
            grad_weights_ = in_cache_.transpose() * grad_out;
            if (bias_)
            {
                grad_biases_ = grad_out.colwise().sum();
            }
        }
        
        // compute gradients for previous layer
        Eigen::MatrixXd grad_in = grad_out * weights_.transpose();

        return grad_in;
    }
}