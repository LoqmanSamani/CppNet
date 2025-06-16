
// layers.hpp file

#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <tuple>


namespace CppNet 
{

    namespace Activations
    {
        class Activation;
        class ReLU;
    }
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
                virtual void update_parameters(Optimizers::Optimizer& optimizer, double learning_rate) = 0;
                virtual ~Layer() = default;
        };
        class Conv2d : public Layer
        {
            public:

                Conv2d(
                    int in_channels,
                    int out_channels,
                    Activations::Activation* activator = nullptr,
                    std::tuple<int, int> kernel_size = std::make_tuple(3, 3),
                    std::tuple<int, int> stride = std::make_tuple(1, 1),
                    std::string padding = "valid",
                    std::tuple<int, int, int, int> num_padding = std::make_tuple(0, 0, 0, 0),
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

                void set_weights(const Eigen::Tensor<double, 4>& weights) { weights_ = weights; }
                void set_biases(const Eigen::Tensor<double, 1>& biases) { biases_ = biases; }

                bool is_trainable() const override { return trainable_; }
                void freeze() { trainable_ = false; }
                void unfreeze() { trainable_ = true; }
                bool has_bias() const { return bias_; }

                void update_parameters(Optimizers::Optimizer& optimizer, double learning_rate) override;
                
                void print_layer_info() const 
                {
                    std::cout << "Layer: " << layer_name_ << std::endl;
                    std::cout << "  Input channels: " << in_channels_ << std::endl;
                    std::cout << "  Output channels: " << out_channels_ << std::endl;
                    std::cout << "  Kernel size: [" << std::get<0>(kernel_size_) << ", " << std::get<1>(kernel_size_) << "]" << std::endl;
                    std::cout << "  Stride: [" << std::get<0>(stride_) << ", " << std::get<1>(stride_) << "]" << std::endl;
                    std::cout << "  Padding: " << padding_ << std::endl;
                    std::cout << "  Trainable: " << (trainable_ ? "Yes" : "No") << std::endl;
                    std::cout << "  Has bias: " << (bias_ ? "Yes" : "No") << std::endl;
                    std::cout << "  Weight shape: [" << weights_.dimension(0) << ", " << weights_.dimension(1) << ", "
                            << weights_.dimension(2) << ", " << weights_.dimension(3) << "]" << std::endl;
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
                std::tuple<int, int> kernel_size_;
                std::tuple<int, int> stride_;
                std::string padding_;
                std::tuple<int, int, int, int> num_padding_;
                std::string padding_mode_;
                std::string layer_name_;
                bool trainable_;
                bool bias_;
                
                Eigen::Tensor<double, 4> weights_;
                Eigen::Tensor<double, 1> biases_;
                Eigen::Tensor<double, 4> grad_weights_;
                Eigen::Tensor<double, 1> grad_biases_;
                Eigen::Tensor<double, 4> in_cache_;
                Eigen::Tensor<double, 4> output_;

                // Input/output dimensions
                int B_, C_, H_, W_; // Input dimensions
                int h_, w_;         // Output dimensions
                
                void init_params_and_grads();
                void init_output();
                Eigen::Tensor<double, 4> pad_input(const Eigen::Tensor<double, 4>& X);
                Eigen::MatrixXd im2col(const Eigen::Tensor<double, 4>& input);
                void col2im_add(const Eigen::MatrixXd& col_matrix, Eigen::Tensor<double, 4>& grad_input);
        };
    
    }
}


#endif // LAYERS_HPP





// layers.cpp file

#include <cmath>
#include <Eigen/Dense>
#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"




namespace CppNet
{
    namespace Layers
    {

        Conv2d::Conv2d(
            int in_channels,
            int out_channels,
            Activations::Activation* activator,
            std::tuple<int, int> kernel_size,
            std::tuple<int, int> stride,
            std::string padding,
            std::tuple<int, int, int, int> num_padding,
            std::string padding_mode,
            std::string layer_name,
            bool trainable,
            bool bias
        ) : 
            in_channels_(in_channels),
            out_channels_(out_channels),
            activator_(activator),
            kernel_size_(kernel_size),
            stride_(stride),
            padding_(padding),
            num_padding_(num_padding),
            padding_mode_(padding_mode),
            layer_name_(layer_name),
            trainable_(trainable),
            bias_(bias)
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
                throw std::runtime_error("Invalid padding mode: " + padding_ + " in layer: " + layer_name_);
            }
            if (padding_mode_ != "zero") 
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

            // handle activator
            if (activator == nullptr) 
            {
                default_relu_ = std::make_unique<Activations::ReLU>();
                activator_ = default_relu_.get();
            }

            // initialize parameters and gradients
            init_params_and_grads();
        }

        void Conv2d::init_params_and_grads()
        {
            // Xavier initialization
            int fan_in = in_channels_ * std::get<0>(kernel_size_) * std::get<1>(kernel_size_);
            int fan_out = out_channels_ * std::get<0>(kernel_size_) * std::get<1>(kernel_size_);
            double scale = std::sqrt(6.0 / (fan_in + fan_out));

            // initialize weights: [out_channels, in_channels, kernel_h, kernel_w]
            weights_ = Eigen::Tensor<double, 4>(out_channels_, in_channels_, std::get<0>(kernel_size_), std::get<1>(kernel_size_));
            weights_.setRandom();
            weights_ = weights_ * scale;

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
            int pad_left = std::get<0>(num_padding_);
            int pad_right = std::get<1>(num_padding_);
            int pad_top = std::get<2>(num_padding_);
            int pad_bottom = std::get<3>(num_padding_);

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

            if (pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0) 
            {
                return X;
            }

            //int padded_h = H_ + pad_top + pad_bottom;
            //int padded_w = W_ + pad_left + pad_right;

            Eigen::array<std::pair<int, int>, 4> paddings = {{
                {0, 0},                    // batch
                {0, 0},                    // channels
                {pad_top, pad_bottom},     // height  
                {pad_left, pad_right}      // width
            }};
            return X.pad(paddings);
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
        }

        Eigen::MatrixXd Conv2d::im2col(const Eigen::Tensor<double, 4>& input)
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

            Eigen::MatrixXd col_matrix(col_height, col_width);
            col_matrix.setZero();

            int col_idx = 0;
            for (int b = 0; b < B_; ++b) 
            {
                for (int oh = 0; oh < out_h; ++oh) 
                {
                    for (int ow = 0; ow < out_w; ++ow) 
                    {
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
                        col_idx++;
                    }
                }
            }

            return col_matrix;
        }

        void Conv2d::col2im_add(const Eigen::MatrixXd& col_matrix, Eigen::Tensor<double, 4>& grad_input)
        {
            const int k_h = std::get<0>(kernel_size_);
            const int k_w = std::get<1>(kernel_size_);
            const int stride_h = std::get<0>(stride_);
            const int stride_w = std::get<1>(stride_);
            const int input_h = grad_input.dimension(2);
            const int input_w = grad_input.dimension(3);

            int col_idx = 0;
            for (int b = 0; b < B_; ++b) 
            {
                for (int oh = 0; oh < h_; ++oh) 
                {
                    for (int ow = 0; ow < w_; ++ow) 
                    {
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
                                        grad_input(b, ic, ih, iw) += col_matrix(row_idx, col_idx);
                                    }
                                    row_idx++;
                                }
                            }
                        }
                        col_idx++;
                    }
                }
            }
        }

        Eigen::Tensor<double, 4> Conv2d::forward(Eigen::Tensor<double, 4>& X)
        {
            // cache input and get dimensions
            in_cache_ = X;
            B_ = X.dimension(0);
            C_ = X.dimension(1);
            H_ = X.dimension(2);
            W_ = X.dimension(3);

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
            Eigen::Tensor<double, 4> input_to_use = pad_input(X);

            // initialize output dimensions
            init_output();

            // im2col transformation
            Eigen::MatrixXd col_matrix = im2col(input_to_use);

            // reshape weights to matrix: [out_channels, k_h * k_w * in_channels]
            const int k_h = std::get<0>(kernel_size_);
            const int k_w = std::get<1>(kernel_size_);
            Eigen::Tensor<double, 2> weight_tensor = weights_.reshape(
                Eigen::array<int, 2>{out_channels_, k_h * k_w * in_channels_}
            );

            Eigen::Map<Eigen::MatrixXd> weight_matrix(
                weight_tensor.data(), 
                out_channels_, 
                k_h * k_w * in_channels_
            );

            Eigen::MatrixXd result = weight_matrix * col_matrix;

            // add bias using broadcasting
            Eigen::Tensor<double, 2> result_tensor = Eigen::TensorMap<Eigen::Tensor<double, 2>>(
                result.data(), out_channels_, B_ * h_ * w_
            );
            if (bias_) 
            {
                Eigen::Tensor<double, 2> bias_broadcasted = biases_.reshape(
                    Eigen::array<int, 2>{out_channels_, 1}
                ).broadcast(Eigen::array<int, 2>{1, B_ * h_ * w_});
                result_tensor += bias_broadcasted;
            }

            // reshape to 4D tensor
            Eigen::Tensor<double, 4> pre_activation = result_tensor.reshape(
                Eigen::array<int, 4>{B_, out_channels_, h_, w_}
            );

            // apply activation
            output_ = activator_->forward(pre_activation);
            return output_;
        }

        Eigen::Tensor<double, 4> Conv2d::backward(Eigen::Tensor<double, 4>& grad_out)
        {
            // validate gradient dimensions
            if (grad_out.dimension(0) != B_ || grad_out.dimension(1) != out_channels_ || 
                grad_out.dimension(2) != h_ || grad_out.dimension(3) != w_) 
            {
                throw std::runtime_error("Gradient output dimensions mismatch in layer: " + layer_name_);
            }

            const int k_h = std::get<0>(kernel_size_);
            const int k_w = std::get<1>(kernel_size_);
            //const int stride_h = std::get<0>(stride_);
            //const int stride_w = std::get<1>(stride_);

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
            Eigen::Tensor<double, 4> grad_pre_activation = activator_->backward(grad_out);

            // Reshape grad_pre_activation to matrix
            Eigen::Tensor<double, 2> grad_out_tensor = grad_pre_activation.reshape(
                Eigen::array<int, 2>{out_channels_, B_ * h_ * w_}
            );
            Eigen::Map<Eigen::MatrixXd> grad_out_matrix(
                grad_out_tensor.data(), 
                out_channels_, 
                B_ * h_ * w_
            );

            if (trainable_)
            {
                Eigen::MatrixXd col_matrix = im2col(input_to_use);
                Eigen::MatrixXd weight_grad_matrix = grad_out_matrix * col_matrix.transpose();
                
                // Convert MatrixXd to Tensor, then reshape
                Eigen::TensorMap<Eigen::Tensor<double, 2>> weight_grad_tensor(
                    weight_grad_matrix.data(),
                    weight_grad_matrix.rows(),
                    weight_grad_matrix.cols()
                );
                
                // Now reshape the tensor
                grad_weights_ = weight_grad_tensor.reshape(
                    Eigen::array<int, 4>{out_channels_, in_channels_, k_h, k_w}
                );
                
                if (bias_)
                {
                    Eigen::VectorXd bias_grad_vec = grad_out_matrix.colwise().sum();

                    Eigen::Tensor<double, 1> bias_grad_tensor(bias_grad_vec.size());
                    for (int i = 0; i < bias_grad_vec.size(); ++i) 
                    {
                        bias_grad_tensor(i) = bias_grad_vec(i);
                    }

                    grad_biases_ = bias_grad_tensor;
                }
            }

            // compute input gradients
            Eigen::Tensor<double, 2> weight_tensor_T = weights_.reshape(
                Eigen::array<int, 2>{k_h * k_w * in_channels_, out_channels_}
            );

            Eigen::Map<Eigen::MatrixXd> weight_matrix_T(
                weight_tensor_T.data(),
                k_h * k_w * in_channels_,
                out_channels_
            );

            Eigen::MatrixXd grad_input_col = weight_matrix_T * grad_out_matrix;

            // convert back to 4D tensor
            col2im_add(grad_input_col, grad_input);

            return grad_input;
        }

        void Conv2d::update_parameters(Optimizers::Optimizer& optimizer, double learning_rate)
        {
            optimizer.update(*this, learning_rate);
        }     
    }
}


// activations.hpp
#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace CppNet
{
    namespace Activations
    {
       
        class Activation
        {
            public:

                virtual Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z) = 0;
                virtual Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da) = 0;
                virtual double forward(double z) = 0;
                virtual Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& z) = 0;
                virtual Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& da) = 0;
                virtual ~Activation() = default;

            protected:

                Eigen::Tensor<double, 2> in_cache_2d_;
                Eigen::Tensor<double, 4> in_cache_4d_;
                Eigen::Tensor<double, 2> sigmoid_cache_2d_; // For Sigmoid
                Eigen::Tensor<double, 4> sigmoid_cache_4d_; // For Sigmoid
        };

       
        class ReLU : public Activation
        {
            public:

                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z) override;
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da) override;
                double forward(double z) override;
                Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& z) override;
                Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& da) override;
        };

       
        class Sigmoid : public Activation
        {
            public:

                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z) override;
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da) override;
                double forward(double z) override;
                Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& z) override;
                Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& da) override;
        };
    }
}

#endif // ACTIVATIONS_HPP


// activation.cpp
#include "activations.hpp"
#include <cmath>

namespace CppNet
{
    namespace Activations
    {
        // ReLU implementations
        Eigen::Tensor<double, 2> ReLU::forward(const Eigen::Tensor<double, 2>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("ReLU: Empty input tensor in 2D forward");
            }
            in_cache_2d_ = z;
            return z.cwiseMax(0.0);
        }

        Eigen::Tensor<double, 2> ReLU::backward(const Eigen::Tensor<double, 2>& da)
        {
            if (da.dimension(0) != in_cache_2d_.dimension(0) ||
                da.dimension(1) != in_cache_2d_.dimension(1) ||
                da.size() == 0)
            {
                throw std::runtime_error("ReLU: Shape mismatch or empty input in 2D backward");
            }
            
            Eigen::Tensor<double, 2> mask = (in_cache_2d_ > 0.0).template cast<double>();
            return da * mask;
        }

        double ReLU::forward(double z)
        {
            return std::max(0.0, z);
        }

        Eigen::Tensor<double, 4> ReLU::forward(const Eigen::Tensor<double, 4>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("ReLU: Empty input tensor in 4D forward");
            }
            in_cache_4d_ = z;
            return z.cwiseMax(0.0);
        }

        Eigen::Tensor<double, 4> ReLU::backward(const Eigen::Tensor<double, 4>& da)
        {
            if (da.dimension(0) != in_cache_4d_.dimension(0) ||
                da.dimension(1) != in_cache_4d_.dimension(1) ||
                da.dimension(2) != in_cache_4d_.dimension(2) ||
                da.dimension(3) != in_cache_4d_.dimension(3) ||
                da.size() == 0)
            {
                throw std::runtime_error("ReLU: Shape mismatch or empty input in 4D backward");
            }
            
            Eigen::Tensor<double, 4> mask = (in_cache_4d_ > 0.0).template cast<double>();
            return da * mask;
        }

        // Sigmoid implementations
        Eigen::Tensor<double, 2> Sigmoid::forward(const Eigen::Tensor<double, 2>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Empty input tensor in 2D forward");
            }
            in_cache_2d_ = z;
            sigmoid_cache_2d_ = (1.0 + (-z).exp()).inverse();
            return sigmoid_cache_2d_;
        }

        Eigen::Tensor<double, 2> Sigmoid::backward(const Eigen::Tensor<double, 2>& da)
        {
            if (da.dimension(0) != in_cache_2d_.dimension(0) ||
                da.dimension(1) != in_cache_2d_.dimension(1) ||
                da.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Shape mismatch or empty input in 2D backward");
            }
            
            return da * sigmoid_cache_2d_ * (1.0 - sigmoid_cache_2d_);
        }

        double Sigmoid::forward(double z)
        {
            double exp_neg_z = std::exp(-z);
            return 1.0 / (1.0 + exp_neg_z);
        }

        Eigen::Tensor<double, 4> Sigmoid::forward(const Eigen::Tensor<double, 4>& z)
        {
            if (z.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Empty input tensor in 4D forward");
            }
            in_cache_4d_ = z;
            sigmoid_cache_4d_ = (1.0 + (-z).exp()).inverse();
            return sigmoid_cache_4d_;
        }

        Eigen::Tensor<double, 4> Sigmoid::backward(const Eigen::Tensor<double, 4>& da)
        {
            if (da.dimension(0) != in_cache_4d_.dimension(0) ||
                da.dimension(1) != in_cache_4d_.dimension(1) ||
                da.dimension(2) != in_cache_4d_.dimension(2) ||
                da.dimension(3) != in_cache_4d_.dimension(3) ||
                da.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Shape mismatch or empty input in 4D backward");
            }
            
            return da * sigmoid_cache_4d_ * (1.0 - sigmoid_cache_4d_);
        }
    }
}

// optimizers.hpp
#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "layers.hpp"

namespace CppNet
{
    namespace Optimizers
    {
       
        class Optimizer
        {
        public:
            virtual void update(CppNet::Layers::Linear& layer, double learning_rate) = 0;
            virtual void update(CppNet::Layers::Conv2d& layer, double learning_rate) = 0;
            virtual ~Optimizer() = default;
        };

        
        class SGD : public Optimizer
        {
        public:
            SGD() = default; // Explicit default constructor
            void update(CppNet::Layers::Linear& layer, double learning_rate) override;
            void update(CppNet::Layers::Conv2d& layer, double learning_rate) override;
        };
    }
}

#endif // OPTIMIZERS_HPP

// optimizers.cpp
#include "optimizers.hpp"



namespace CppNet
{
    namespace Optimizers
    {
        void SGD::update(CppNet::Layers::Linear& layer, double learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }

            // validate dimensions
            const auto& weights = layer.get_weights();
            const auto& grad_weights = layer.get_grad_weights();
            if (weights.dimensions() != grad_weights.dimensions())
            {
                throw std::runtime_error("Weight and gradient dimension mismatch in Linear layer: " + layer.get_layer_name());
            }

            layer.get_weights() -= learning_rate * layer.get_grad_weights();

            if (layer.has_bias())
            {
                const auto& biases = layer.get_biases();
                const auto& grad_biases = layer.get_grad_biases();
                if (biases.dimensions() != grad_biases.dimensions())
                {
                    throw std::runtime_error("Bias and gradient dimension mismatch in Linear layer: " + layer.get_layer_name());
                }
                layer.get_biases() -= learning_rate * layer.get_grad_biases();
            }
        }

        void SGD::update(CppNet::Layers::Conv2d& layer, double learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }

            // validate dimensions
            const auto& weights = layer.get_weights();
            const auto& grad_weights = layer.get_grad_weights();
            if (weights.dimensions() != grad_weights.dimensions())
            {
                throw std::runtime_error("Weight and gradient dimension mismatch in Conv2d layer: " + layer.get_layer_name());
            }

            layer.get_weights() -= learning_rate * layer.get_grad_weights();

            if (layer.has_bias())
            {
                const auto& biases = layer.get_biases();
                const auto& grad_biases = layer.get_grad_biases();
                if (biases.dimensions() != grad_biases.dimensions())
                {
                    throw std::runtime_error("Bias and gradient dimension mismatch in Conv2d layer: " + layer.get_layer_name());
                }
                layer.get_biases() -= learning_rate * layer.get_grad_biases();
            }
        }
    }
}