#include <Eigen/Dense>
#include "layers.hpp"

namespace CppNet
{
    // a simple implementation of a linear(dense) layer  
    Linear::Linear(int in_size, int out_size) : in_size_(in_size), out_size_(out_size)
    {
        // initialize parameters
        init_params();
        // initialize gradient matrices with zero
        grad_weights_ = Eigen::MatrixXd::Zero(in_size_, out_size_);
        grad_biases_ = Eigen::VectorXd::Zero(out_size_);
    } 

    void Linear::init_params()
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        // initializing weights with Xavier method
        double scale = std::sqrt(6.0 / (in_size_ + out_size_));
        std::uniform_real_distribution<> dis(-scale, scale);

        weights_ = Eigen::MatrixXd(in_size_, out_size_);
        for (int i = 0; i < in_size_; i++)
        {
            for (int j = 0; j < out_size_; j++)
            {
                weights_(i, j) = dis(gen);
            }
        }

        // initialize biases with zero
        biases_ = Eigen::VectorXd::Zero(out_size_);        

    }

    Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd& X)
    {
        // store x to use in gradient calculation
        in_cache_ = X;

        Eigen::MatrixXd output = X * weights_;
        output.rowwise() += biases_.transpose();

        return output;

    }

    Eigen::MatrixXd Linear::backward(const Eigen::MatrixXd& grad_out)
    {
        // compute weights and biases gradients
        grad_weights_ = in_cache_.transpose() * grad_out;
        grad_biases_ = grad_out.colwise().sum(); 

        // compute gradients for previos layer
        Eigen::MatrixXd grad_in = grad_out * weights_.transpose();

        return grad_in;

    }

    void Linear::update_params(double lr)
    {
        // update weights and biases 
        // but if we use a different optimizer!!
        weights_ -= lr * grad_weights_;
        biases_ -= lr * grad_weights_;

    }
    
}