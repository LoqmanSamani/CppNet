#include <Eigen/Dense>
#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"





namespace CppNet
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
            throw std::runtime_error("Shape mismatch: in layer: " + layer_name_ + " X.cols() must be equal weights_.rows()!");
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
            throw std::runtime_error("Shape mismatch: in layer: " + layer_name_ + " grad_out.rows() must be equal in_cache_.rows()!");
        }
        if (grad_out.cols() != out_size_)
        {
            throw std::runtime_error("Shape mismatch: in layer: " + layer_name_ + " grad_out.cols() must be equal out_size_!");
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

    Conv2d::Conv2d(
        int in_channels, int out_channels, std::tuple<int, int> kernel_size, std::tuple<int, int> stride, std::string padding,
        std::tuple<int, int> num_padding, std::string padding_mode, std::string layer_name, bool trainable, bool bias)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride),
        padding_(padding), num_padding_(num_padding), padding_mode_(padding_mode), layer_name_(layer_name),
        trainable_(trainable), bias_(bias)
        {
            // check if in and out sizes are positive integers
            if (in_channels <= 0 || out_channels <= 0)
            {
                throw std::runtime_error("in_chnnels and out_channels of layer: " + layer_name + " must be positive integers!");
            }

            // initialize filters, biases and gradients 
            init_params_and_grads();   
        }

    Eigen::Tensor<double, 4> Conv2d::forward(Eigen::Tensor<double, 4>& X)
    {
        B_ = X.dimension(0); // batch
        C_ = X.dimension(1); // channels
        H_ = X.dimension(2);
        W_ = X.dimension(3);

        const int pad_h = std::get<2>(num_padding_);
        const int pad_w = std::get<0>(num_padding_);
        const int K_h = std::get<0>(kernel_size_);
        const int K_w = std::get<1>(kernel_size_);
        const int stride_h = std::get<0>(stride_);
        const int stride_w = std::get<1>(stride_);

        // output dims
        //int H_out = (H_ + 2 * pad_h - K_h) / stride_h + 1;
        //int W_out = (W_ + 2 * pad_w - K_w) / stride_w + 1;

        // pad input
        Eigen::Tensor<double, 4> pad_x = pad_input(X); // shape: [B_, C_, h_, w_]

        // output tensor
        Eigen::Tensor<double, 4> output(B_, out_channels_, h_, w_);
        output.setZero();

        ReLU relu;

        for (int b = 0; b < B_; ++b)
        {
            for (int oc = 0; oc < out_channels_; ++oc)
            {
                Eigen::Tensor<double, 2> z(h_, w_);
                z.setZero();

                for (int ic = 0; ic < C_; ++ic)
                {
                    for (int h = 0; h < h_; ++h)
                    {
                        int h_start = h * stride_h;
                        for (int w = 0; w < w_; ++w)
                        {
                            int w_start = w * stride_w;

                            // extract patch from input
                            Eigen::array<Eigen::Index, 4> x_offsets = {b, ic, h_start, w_start};
                            Eigen::array<Eigen::Index, 4> x_extents = {1, 1, K_h, K_w};
                            auto patch = pad_x.slice(x_offsets, x_extents).reshape(Eigen::array<Eigen::Index, 2>{K_h, K_w});

                            // extract corresponding filter
                            Eigen::array<Eigen::Index, 4> w_offsets = {oc, ic, 0, 0};
                            Eigen::array<Eigen::Index, 4> w_extents = {1, 1, K_h, K_w};
                            auto filter = weights_.slice(w_offsets, w_extents).reshape(Eigen::array<Eigen::Index, 2>{K_h, K_w});

                            // wlement-wise multiply and accumulate
                            double conv_val = (patch * filter).sum();
                            z(h, w) += conv_val;
                        }
                    }
                }

                // apply relu and write to output
                auto z_relu = relu.forward(z).reshape(Eigen::array<Eigen::Index, 3>{1, h_, w_});
                Eigen::array<Eigen::Index, 4> out_offsets = {b, oc, 0, 0};
                Eigen::array<Eigen::Index, 4> out_extents = {1, 1, h_, w_};
                output.slice(out_offsets, out_extents) = z_relu;
            }
        }

        return output;
    }

    void Conv2d::init_output()
    {

        // TODO: methods for "valid" & "same" should be implemented.
        // padding='valid' is the same as no padding. padding='same' pads the input so the output has the shape as the input.
        // However, this mode doesn’t support any stride values other than 1.

        // compute output shape if padding_is none (not valid or same)
        if (padding_ == "none")
        {
            double h_f = (H_ + 2.0 * std::get<0>(num_padding_) - static_cast<double>(std::get<0>(kernel_size_)) + 1.0) / static_cast<double>(std::get<0>(stride_));
            double w_f = (W_ + 2.0 * std::get<1>(num_padding_) - static_cast<double>(std::get<1>(kernel_size_)) + 1.0) / static_cast<double>(std::get<1>(stride_));

            h_ = static_cast<int>(std::floor(h_f));
            w_ = static_cast<int>(std::floor(w_f));
        }
        // initialize output tensor with zeros and computed shape
        output_.resize(B_, C_, h_, w_);
        output_.setZero();

    }

    void Conv2d::init_params_and_grads()
    {
        weights_.resize(out_channels_, in_channels_, std::get<0>(kernel_size_), std::get<1>(kernel_size_));
        grad_weights_.resize(out_channels_, in_channels_, std::get<0>(kernel_size_), std::get<1>(kernel_size_));

        std::random_device rd;
        std::mt19937 gen(rd());

        double scale = std::sqrt(6.0 / (in_channels_ + out_channels_));
        std::uniform_real_distribution<> dis(-scale, scale);

        // initializing weights with Xavier method (default method).
        weights_.generate([&]() { return dis(gen); });
        // initialize weight-gradient matrix with zero
        grad_weights_.setZero();

        if (bias_)
        {
            // initialize biases with zero, if bias_ is true.
            biases_ = Eigen::VectorXd::Zero(out_channels_); 
            // initialize bias-gradient matrix with zero 
            grad_biases_ = Eigen::VectorXd::Zero(out_channels_);
        }
        else
        {
            // initialize empty biases and gradients when bias is false
            biases_ = Eigen::VectorXd(0);
            grad_biases_ = Eigen::VectorXd(0);
        }
    }

    Eigen::Tensor<double, 4> Conv2d::pad_input(Eigen::Tensor<double, 4>& X)
    {
        Eigen::Tensor<double, 4> pad_x(B_, C_, H_, W_);
        pad_x.setZero();

        // TODO: implement other methods
        if (padding_mode_ == "zero")
        {
            
            for (int i = 0; i < B_; i++)
            {
                for (int j = 0; j < C_; j++)
                {
                    for (int h = std::get<2>(num_padding_); h < X.dimension(2) - std::get<2>(num_padding_) - std::get<3>(num_padding_); h++)
                    {
                        for (int w = std::get<0>(num_padding_); w < X.dimension(3) - std::get<0>(num_padding_) - std::get<1>(num_padding_); w++)
                        {
                            pad_x(i, j, h, w) = X(i, j, h, w);
                        }
                    }
                }
            }

        }
        return pad_x;
    }

    Eigen::Tensor<double, 4> Conv2d::backward(Eigen::Tensor<double, 4>& grad_out)
    {

    }

}