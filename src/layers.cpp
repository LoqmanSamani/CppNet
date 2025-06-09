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

        void Linear::update_parameters(Optimizers::Optimizer& optimizer, double learning_rate)
        {
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
                Eigen::array<Eigen::Index, 2> broadcast_dims({X.dimension(0), 1});
                Eigen::Tensor<double, 2> bias_broadcasted = biases_.reshape(Eigen::array<Eigen::Index, 2>({1, biases_.dimension(0)})).broadcast(broadcast_dims);
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


        Conv2d::Conv2d(
            int in_channels, int out_channels, Activations::Activation* activator, 
            std::tuple<int, int> kernel_size, std::tuple<int, int> stride, 
            std::string padding, std::tuple<int, int, int, int> num_padding, 
            std::string padding_mode, std::string layer_name, bool trainable, bool bias): 
            in_channels_(in_channels), out_channels_(out_channels), 
            kernel_size_(kernel_size), stride_(stride), padding_(padding), 
            num_padding_(num_padding), padding_mode_(padding_mode), 
            layer_name_(layer_name), trainable_(trainable), bias_(bias)
        {
            if (in_channels <= 0 || out_channels <= 0)
            {
                throw std::runtime_error("in_channels and out_channels of layer: " + layer_name + " must be positive integers!");
            }

            if (layer_name.empty()) 
            {
                layer_name_ = "Conv2D_" + std::to_string(in_channels_) + "x" + std::to_string(out_channels_);
            }

            // handle activator
            if (activator == nullptr) {
                default_relu_ = std::make_unique<Activations::ReLU>();
                activator_ = default_relu_.get();
            } else {
                activator_ = activator;
            }

            // initialize parameters and gradients
            init_params_and_grads();
        }

        void Conv2d::init_params_and_grads()
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            // Xavier initialization
            int fan_in = in_channels_ * std::get<0>(kernel_size_) * std::get<1>(kernel_size_);
            int fan_out = out_channels_ * std::get<0>(kernel_size_) * std::get<1>(kernel_size_);
            double scale = std::sqrt(6.0 / (fan_in + fan_out));
            std::uniform_real_distribution<> dis(-scale, scale);

            // initialize weights: [out_channels, in_channels, kernel_h, kernel_w]
            weights_ = Eigen::Tensor<double, 4>(out_channels_, in_channels_, std::get<0>(kernel_size_), std::get<1>(kernel_size_));

            for (int oc = 0; oc < out_channels_; ++oc)
            {
                for (int ic = 0; ic < in_channels_; ++ic)
                {
                    for (int kh = 0; kh < std::get<0>(kernel_size_); ++kh)
                    {
                        for (int kw = 0; kw < std::get<1>(kernel_size_); ++kw)
                        {
                            weights_(oc, ic, kh, kw) = dis(gen);
                        }
                    }
                }
            }

            // initialize gradients
            grad_weights_ = Eigen::Tensor<double, 4>(out_channels_, in_channels_, std::get<0>(kernel_size_), std::get<1>(kernel_size_));
            grad_weights_.setZero();

            if (bias_)
            {
                biases_ = Eigen::Tensor<double, 1>(out_channels_);
                biases_.setZero(); 
                grad_biases_ = Eigen::Tensor<double, 1>(out_channels_);
                grad_biases_.setZero(); 
            }
            else
            {
                biases_ = Eigen::Tensor<double, 1>(0);
                grad_biases_ = Eigen::Tensor<double, 1>(0);
            }
        }

        Eigen::Tensor<double, 4> Conv2d::pad_input(const Eigen::Tensor<double, 4>& X)
        {
            // num_padding_ = (left, right, top, bottom)
            int pad_left = std::get<0>(num_padding_);
            int pad_right = std::get<1>(num_padding_);
            int pad_top = std::get<2>(num_padding_);
            int pad_bottom = std::get<3>(num_padding_);
            
            int padded_h = H_ + pad_top + pad_bottom;
            int padded_w = W_ + pad_left + pad_right;
            
            Eigen::Tensor<double, 4> padded_x(B_, C_, padded_h, padded_w);
            
            if (padding_mode_ == "zero")
            {
                padded_x.setZero();
                
                // copy original data to the center of padded tensor
                for (int b = 0; b < B_; ++b)
                {
                    for (int c = 0; c < C_; ++c)
                    {
                        for (int h = 0; h < H_; ++h)
                        {
                            for (int w = 0; w < W_; ++w)
                            {
                                padded_x(b, c, h + pad_top, w + pad_left) = X(b, c, h, w);
                            }
                        }
                    }
                }
            }
            
            return padded_x;
        }

        void Conv2d::init_output()
        {
            if (padding_ == "valid" || padding_ == "none")
            {
                // calculate output dimensions
                int pad_h = std::get<2>(num_padding_) + std::get<3>(num_padding_); // top + bottom
                int pad_w = std::get<0>(num_padding_) + std::get<1>(num_padding_); // left + right
                
                h_ = (H_ + pad_h - std::get<0>(kernel_size_)) / std::get<0>(stride_) + 1;
                w_ = (W_ + pad_w - std::get<1>(kernel_size_)) / std::get<1>(stride_) + 1;
            }
            else if (padding_ == "same")
            {
                // for "same" padding, output size equals input size (when stride=1)
                h_ = (H_ + std::get<0>(stride_) - 1) / std::get<0>(stride_);
                w_ = (W_ + std::get<1>(stride_) - 1) / std::get<1>(stride_);
            }

            // initialize output tensor
            output_ = Eigen::Tensor<double, 4>(B_, out_channels_, h_, w_);
            output_.setZero();
        }

        void Conv2d::update_parameters(Optimizers::Optimizer& optimizer, double learning_rate)
        {
            optimizer.update(*this, learning_rate);
        }

        Eigen::Tensor<double, 4> Conv2d::forward(Eigen::Tensor<double, 4>& X)
        {
            // cache input for backward pass
            in_cache_ = X;
            
            // get input dimensions
            B_ = X.dimension(0); // batch
            C_ = X.dimension(1); // channels  
            H_ = X.dimension(2); // height
            W_ = X.dimension(3); // width

            // validate input channels
            if (C_ != in_channels_) 
            {
                throw std::runtime_error("Input channels mismatch in layer: " + layer_name_);
            }

            const int k_h = std::get<0>(kernel_size_);
            const int k_w = std::get<1>(kernel_size_);
            const int stride_h = std::get<0>(stride_);
            const int stride_w = std::get<1>(stride_);

            // pad input if needed
            Eigen::Tensor<double, 4> input_to_use = X;
            if (padding_ != "valid" && (std::get<0>(num_padding_) > 0 || std::get<1>(num_padding_) > 0 || std::get<2>(num_padding_) > 0 || std::get<3>(num_padding_) > 0)) 
            {
                input_to_use = pad_input(X);
            }

            // initialize output tensor
            init_output();

            // perform convolution
            for (int b = 0; b < B_; ++b)
            {
                for (int oc = 0; oc < out_channels_; ++oc)
                {
                    for (int oh = 0; oh < h_; ++oh)
                    {
                        for (int ow = 0; ow < w_; ++ow)
                        {
                            double conv_sum = 0.0;
                            
                            // convolution over all input channels
                            for (int ic = 0; ic < in_channels_; ++ic)
                            {
                                for (int kh = 0; kh < k_h; ++kh)
                                {
                                    for (int kw = 0; kw < k_w; ++kw)
                                    {
                                        int ih = oh * stride_h + kh;
                                        int iw = ow * stride_w + kw;
                                        
                                        // bounds checking
                                        if (ih >= 0 && ih < input_to_use.dimension(2) && iw >= 0 && iw < input_to_use.dimension(3))
                                        {
                                            conv_sum += input_to_use(b, ic, ih, iw) * weights_(oc, ic, kh, kw);
                                        }
                                    }
                                }
                            }
                            
                            // add bias if enabled
                            if (bias_) 
                            {
                                conv_sum += biases_(oc);
                            }

                            // apply activation and store result
                            output_(b, oc, oh, ow) = activator_->forward(conv_sum);
                        }
                    }
                }
            }

            return output_;
        }

        Eigen::Tensor<double, 4> Conv2d::backward(Eigen::Tensor<double, 4>& grad_out)
        {
            // validate input gradient dimensions
            if (grad_out.dimension(0) != B_ || grad_out.dimension(1) != out_channels_ || grad_out.dimension(2) != h_ || grad_out.dimension(3) != w_)
            {
                throw std::runtime_error("Gradient output dimensions mismatch in layer: " + layer_name_);
            }

            // get kernel and stride parameters
            const int k_h = std::get<0>(kernel_size_);
            const int k_w = std::get<1>(kernel_size_);
            const int stride_h = std::get<0>(stride_);
            const int stride_w = std::get<1>(stride_);
            
            // initialize gradients (only if trainable)
            if (trainable_)
            {
                grad_weights_.setZero();
                if (bias_)
                {
                    grad_biases_.setZero();
                }
            }
            
            // initialize input gradient
            Eigen::Tensor<double, 4> grad_input(in_cache_.dimensions());
            grad_input.setZero();
            
            // get the input used in forward pass (potentially padded)
            Eigen::Tensor<double, 4> input_to_use = in_cache_;
            if (padding_ != "valid" && (std::get<0>(num_padding_) > 0 || std::get<1>(num_padding_) > 0 || 
                                        std::get<2>(num_padding_) > 0 || std::get<3>(num_padding_) > 0))
            {
                input_to_use = pad_input(in_cache_);
            }
            
            // apply activation backward pass
            // we need to reconstruct the pre-activation values and apply activation derivative
            Eigen::Tensor<double, 4> grad_pre_activation(grad_out.dimensions());
            
            // for each output position, compute the activation derivative
            for (int b = 0; b < B_; ++b)
            {
                for (int oc = 0; oc < out_channels_; ++oc)
                {
                    for (int oh = 0; oh < h_; ++oh)
                    {
                        for (int ow = 0; ow < w_; ++ow)
                        {
                            // reconstruct pre-activation value
                            double pre_activation = 0.0;
                            
                            for (int ic = 0; ic < in_channels_; ++ic)
                            {
                                for (int kh = 0; kh < k_h; ++kh)
                                {
                                    for (int kw = 0; kw < k_w; ++kw)
                                    {
                                        int ih = oh * stride_h + kh;
                                        int iw = ow * stride_w + kw;
                                        
                                        if (ih >= 0 && ih < input_to_use.dimension(2) && iw >= 0 && iw < input_to_use.dimension(3))
                                        {
                                            pre_activation += input_to_use(b, ic, ih, iw) * weights_(oc, ic, kh, kw);
                                        }
                                    }
                                }
                            }
                            
                            if (bias_)
                            {
                                pre_activation += biases_(oc);
                            }
                            // TODO: add derivative of other activation functions!
                            // suppose we use only ReLU
                            // apply activation backward (ReLU derivative: 1 if x > 0, else 0)
                            double activation_grad = (pre_activation > 0.0) ? 1.0 : 0.0;
                            grad_pre_activation(b, oc, oh, ow) = grad_out(b, oc, oh, ow) * activation_grad;
                        }
                    }
                }
            }
            
            // compute gradients w.r.t. weights, biases, and input
            for (int b = 0; b < B_; ++b)
            {
                for (int oc = 0; oc < out_channels_; ++oc)
                {
                    for (int oh = 0; oh < h_; ++oh)
                    {
                        for (int ow = 0; ow < w_; ++ow)
                        {
                            double grad_output_elem = grad_pre_activation(b, oc, oh, ow);
                            
                            // gradient w.r.t. bias (only if trainable and has bias)
                            if (trainable_ && bias_)
                            {
                                grad_biases_(oc) += grad_output_elem;
                            }
                            
                            // gradient w.r.t. weights and input
                            for (int ic = 0; ic < in_channels_; ++ic)
                            {
                                for (int kh = 0; kh < k_h; ++kh)
                                {
                                    for (int kw = 0; kw < k_w; ++kw)
                                    {
                                        int ih = oh * stride_h + kh;
                                        int iw = ow * stride_w + kw;
                                        
                                        if (ih >= 0 && ih < input_to_use.dimension(2) && 
                                            iw >= 0 && iw < input_to_use.dimension(3))
                                        {
                                            // gradient w.r.t. weights (only if trainable)
                                            if (trainable_)
                                            {
                                                grad_weights_(oc, ic, kh, kw) += input_to_use(b, ic, ih, iw) * grad_output_elem;
                                            }
                                            
                                            // gradient w.r.t. input
                                            // map back to original input coordinates if padding was applied
                                            int orig_ih = ih;
                                            int orig_iw = iw;
                                            
                                            if (padding_ != "valid")
                                            {
                                                orig_ih = ih - std::get<2>(num_padding_); // subtract top padding
                                                orig_iw = iw - std::get<0>(num_padding_); // subtract left padding
                                            }
                                            
                                            // check if coordinates are valid for original input
                                            if (orig_ih >= 0 && orig_ih < in_cache_.dimension(2) && orig_iw >= 0 && orig_iw < in_cache_.dimension(3))
                                            {
                                                grad_input(b, ic, orig_ih, orig_iw) += weights_(oc, ic, kh, kw) * grad_output_elem;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            return grad_input;
        }

}