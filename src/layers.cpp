#include <cmath>
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

            // compute weight gradients
            //if (trainable_) 
            //{
            //    Eigen::MatrixXd col_matrix = im2col(input_to_use);
            //    Eigen::MatrixXd weight_grad_matrix = grad_out_matrix * col_matrix.transpose();
            //   grad_weights_ = weight_grad_matrix.reshape(
            //        Eigen::array<int, 4>{out_channels_, in_channels_, k_h, k_w}
            //    );

            //    if (bias_) 
            //    {
            //        grad_biases_ = grad_out_matrix.colwise().sum();
            //    }
            //}
            // compute weight gradients
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