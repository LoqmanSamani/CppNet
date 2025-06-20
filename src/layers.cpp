#include <cmath>
#include <Eigen/Dense>
#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"




namespace CppNet
{
    namespace Layers
    {
        /************************************** Linear/Dense *******************************************/

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

        /************************************** Conv2d *******************************************/    

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

        /************************************** MaxPool2D *******************************************/  

        MaxPool2D::MaxPool2D(
            std::tuple<int, int> kernel_size,
            std::tuple<int, int> stride,
            std::string padding,
            std::tuple<int, int, int, int> num_padding,
            std::string padding_mode,
            std::string layer_name
        ) :
            kernel_size_(kernel_size),
            stride_(stride),
            padding_(padding),
            num_padding_(num_padding),
            padding_mode_(padding_mode),
            layer_name_(layer_name)
        {
            // Input validation
            if (std::get<0>(kernel_size_) <= 0 || std::get<1>(kernel_size_) <= 0) {
                throw std::runtime_error("Kernel size must be positive in layer: " + layer_name_);
            }
            if (std::get<0>(stride_) <= 0 || std::get<1>(stride_) <= 0) {
                throw std::runtime_error("Stride must be positive in layer: " + layer_name_);
            }
            if (padding_ != "valid" && padding_ != "same") { // Fixed: "same" instead of "none"
                throw std::runtime_error("Invalid padding mode: " + padding_ + " in layer: " + layer_name_);
            }
            if (padding_mode_ != "zero") {
                throw std::runtime_error("Invalid padding mode: " + padding_mode_ + " in layer: " + layer_name_);
            }
            if (std::get<0>(num_padding_) < 0 || std::get<1>(num_padding_) < 0 || 
                std::get<2>(num_padding_) < 0 || std::get<3>(num_padding_) < 0) {
                throw std::runtime_error("Padding values must be non-negative in layer: " + layer_name_);
            }
        }

        Eigen::Tensor<double, 4> MaxPool2D::pad_input(const Eigen::Tensor<double, 4>& X)
        {
            if (padding_ == "valid") {
                return X;   
            }
            
            int k_h = std::get<0>(kernel_size_);
            int k_w = std::get<1>(kernel_size_);
            int stride_h = std::get<0>(stride_);
            int stride_w = std::get<1>(stride_);
            
            int pad_left, pad_right, pad_top, pad_bottom;
            
            if (padding_ == "same") {
                // Compute padding to maintain output size for "same" padding
                int output_h = (H_ + stride_h - 1) / stride_h; 
                int output_w = (W_ + stride_w - 1) / stride_w; 
                int pad_h_total = std::max(0, (output_h - 1) * stride_h + k_h - H_);
                int pad_w_total = std::max(0, (output_w - 1) * stride_w + k_w - W_);

                pad_top = pad_h_total / 2;
                pad_bottom = pad_h_total - pad_top;
                pad_left = pad_w_total / 2;
                pad_right = pad_w_total - pad_left;
            } else {
                // Use explicit padding values
                pad_left = std::get<0>(num_padding_);
                pad_right = std::get<1>(num_padding_);
                pad_top = std::get<2>(num_padding_);
                pad_bottom = std::get<3>(num_padding_);
            }
            
            if (pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0) {
                return X;
            }
            
            Eigen::array<std::pair<int, int>, 4> paddings = {{
                {0, 0},                    // batch
                {0, 0},                    // channels
                {pad_top, pad_bottom},     // height  
                {pad_left, pad_right}      // width
            }};
            
            return X.pad(paddings);
        }

        void MaxPool2D::init_output()
        {
            int k_h = std::get<0>(kernel_size_);
            int k_w = std::get<1>(kernel_size_);
            int stride_h = std::get<0>(stride_);
            int stride_w = std::get<1>(stride_);
            
            if (padding_ == "valid") {
                h_ = (H_ - k_h) / stride_h + 1;
                w_ = (W_ - k_w) / stride_w + 1;
            } else if (padding_ == "same") {
                h_ = (H_ + stride_h - 1) / stride_h;
                w_ = (W_ + stride_w - 1) / stride_w;
            } else {
                // Custom padding
                int pad_h = std::get<2>(num_padding_) + std::get<3>(num_padding_);
                int pad_w = std::get<0>(num_padding_) + std::get<1>(num_padding_);
                h_ = (H_ + pad_h - k_h) / stride_h + 1;
                w_ = (W_ + pad_w - k_w) / stride_w + 1;
            }

            if (h_ <= 0 || w_ <= 0) {
                throw std::runtime_error("Invalid output dimensions in layer: " + layer_name_);
            }

            output_ = Eigen::Tensor<double, 4>(B_, C_, h_, w_);
        }

        Eigen::Tensor<double, 4> MaxPool2D::forward(const Eigen::Tensor<double, 4>& X)
        {
            B_ = X.dimension(0);
            C_ = X.dimension(1);
            H_ = X.dimension(2);
            W_ = X.dimension(3);
            
            // Validate input
            if (H_ < std::get<0>(kernel_size_) || W_ < std::get<1>(kernel_size_)) {
                throw std::runtime_error("Input dimensions too small for kernel in layer: " + layer_name_);
            }
            
            // Pad input if needed
            Eigen::Tensor<double, 4> input_to_use = pad_input(X);
            in_cache_ = input_to_use; // Cache padded input
            
            // Initialize output dimensions
            init_output();
            
            // Get kernel and stride parameters
            int kh = std::get<0>(kernel_size_);
            int kw = std::get<1>(kernel_size_);
            int sh = std::get<0>(stride_);
            int sw = std::get<1>(stride_);

            // Apply max pooling
            for (int b = 0; b < B_; ++b) {
                for (int c = 0; c < C_; ++c) {
                    for (int oh = 0; oh < h_; ++oh) {
                        for (int ow = 0; ow < w_; ++ow) {
                            // Calculate input region bounds
                            int h_start = oh * sh;
                            int w_start = ow * sw;
                            int h_end = std::min(h_start + kh, static_cast<int>(input_to_use.dimension(2)));
                            int w_end = std::min(w_start + kw, static_cast<int>(input_to_use.dimension(3)));
                            
                            // Find maximum in the kernel window
                            double max_val = -std::numeric_limits<double>::infinity();
                            for (int kh_idx = h_start; kh_idx < h_end; ++kh_idx) {
                                for (int kw_idx = w_start; kw_idx < w_end; ++kw_idx) {
                                    max_val = std::max(max_val, input_to_use(b, c, kh_idx, kw_idx));
                                }
                            }
                            
                            output_(b, c, oh, ow) = max_val;
                        }
                    }
                }
            }
            
            return output_;
        }

        Eigen::Tensor<double, 4> MaxPool2D::backward(const Eigen::Tensor<double, 4>& grad_out)
        {
            // Initialize gradient tensor with same dimensions as cached input
            Eigen::Tensor<double, 4> grad_input_padded = Eigen::Tensor<double, 4>(
                in_cache_.dimension(0), in_cache_.dimension(1), 
                in_cache_.dimension(2), in_cache_.dimension(3)
            );
            grad_input_padded.setZero();
            
            int kh = std::get<0>(kernel_size_);
            int kw = std::get<1>(kernel_size_);
            int sh = std::get<0>(stride_);
            int sw = std::get<1>(stride_);
            
            // Iterate through output positions
            for (int b = 0; b < B_; ++b) {
                for (int c = 0; c < C_; ++c) {
                    for (int oh = 0; oh < h_; ++oh) {
                        for (int ow = 0; ow < w_; ++ow) {
                            // Calculate input region bounds
                            int h_start = oh * sh;
                            int w_start = ow * sw;
                            int h_end = std::min(h_start + kh, static_cast<int>(in_cache_.dimension(2)));
                            int w_end = std::min(w_start + kw, static_cast<int>(in_cache_.dimension(3)));
                            
                            // Find the position of maximum value in the kernel window
                            double max_val = -std::numeric_limits<double>::infinity();
                            int max_h = h_start;
                            int max_w = w_start;
                            
                            for (int kh_idx = h_start; kh_idx < h_end; ++kh_idx) {
                                for (int kw_idx = w_start; kw_idx < w_end; ++kw_idx) {
                                    if (in_cache_(b, c, kh_idx, kw_idx) > max_val) {
                                        max_val = in_cache_(b, c, kh_idx, kw_idx);
                                        max_h = kh_idx;
                                        max_w = kw_idx;
                                    }
                                }
                            }
                            
                            // Propagate gradient to the max position
                            grad_input_padded(b, c, max_h, max_w) += grad_out(b, c, oh, ow);
                        }
                    }
                }
            }
            
            // Remove padding from gradient if input was padded
            if (padding_ == "valid" || 
                (std::get<0>(num_padding_) == 0 && std::get<1>(num_padding_) == 0 && 
                std::get<2>(num_padding_) == 0 && std::get<3>(num_padding_) == 0)) {
                return grad_input_padded;
            }
            
            // Extract the original input region from padded gradient
            int pad_top = (padding_ == "same") ? 0 : std::get<2>(num_padding_);
            int pad_left = (padding_ == "same") ? 0 : std::get<0>(num_padding_);
            
            if (padding_ == "same") {
                // Calculate padding for "same" mode
                int k_h = std::get<0>(kernel_size_);
                int k_w = std::get<1>(kernel_size_);
                int stride_h = std::get<0>(stride_);
                int stride_w = std::get<1>(stride_);
                
                int output_h = (H_ + stride_h - 1) / stride_h;
                int output_w = (W_ + stride_w - 1) / stride_w;
                int pad_h_total = std::max(0, (output_h - 1) * stride_h + k_h - H_);
                int pad_w_total = std::max(0, (output_w - 1) * stride_w + k_w - W_);
                
                pad_top = pad_h_total / 2;
                pad_left = pad_w_total / 2;
            }
            
            Eigen::Tensor<double, 4> grad_input(B_, C_, H_, W_);
            
            for (int b = 0; b < B_; ++b) {
                for (int c = 0; c < C_; ++c) {
                    for (int h = 0; h < H_; ++h) {
                        for (int w = 0; w < W_; ++w) {
                            grad_input(b, c, h, w) = grad_input_padded(b, c, h + pad_top, w + pad_left);
                        }
                    }
                }
            }
            
            return grad_input;
        }

        /************************************** Flatten *******************************************/ 

        Flatten::Flatten(int start_dim, int end_dim, std::string layer_name) :
            start_dim_(start_dim), end_dim_(end_dim), layer_name_(layer_name),
            in_size_(0), out_size_(0), input_rank_(0) 
        {
            if (start_dim < 0) {
                throw std::invalid_argument("Flatten: start_dim must be non-negative");
            }
        }

        Eigen::Tensor<double, 2> Flatten::forward(const Eigen::Tensor<double, 4>& X) {
            input_rank_ = 4;
            if (X.size() == 0) {
                throw std::runtime_error("Flatten: Empty input tensor");
            }

            // Store input shape
            in_shape_.resize(4);
            in_size_ = 1;
            for (int i = 0; i < 4; ++i) {
                in_shape_[i] = X.dimension(i);
                in_size_ *= X.dimension(i);
            }

            // Resolve end_dim
            int resolved_end_dim = end_dim_ < 0 ? 3 : end_dim_;
            if (start_dim_ >= 4 || resolved_end_dim >= 4 || start_dim_ > resolved_end_dim) {
                throw std::invalid_argument("Flatten: Invalid start_dim or end_dim for 4D tensor");
            }

            // Common case: keep batch dimension, flatten the rest
            if (start_dim_ == 1 && resolved_end_dim == 3) {
                int batch_size = X.dimension(0);
                int feature_size = X.dimension(1) * X.dimension(2) * X.dimension(3);
                out_size_ = feature_size;
                
                Eigen::array<int, 2> out_dims = {batch_size, feature_size};
                return X.reshape(out_dims);
            }
            
            // General case
            int first_dim = 1;
            for (int i = 0; i < start_dim_; ++i) {
                first_dim *= X.dimension(i);
            }
            
            int second_dim = 1;
            for (int i = start_dim_; i <= resolved_end_dim; ++i) {
                second_dim *= X.dimension(i);
            }
            
            for (int i = resolved_end_dim + 1; i < 4; ++i) {
                second_dim *= X.dimension(i);
            }
            
            out_size_ = second_dim;
            Eigen::array<int, 2> out_dims = {first_dim, second_dim};
            return X.reshape(out_dims);
        }

        Eigen::Tensor<double, 2> Flatten::forward(const Eigen::Tensor<double, 3>& X) {
            input_rank_ = 3;
            // Similar implementation for 3D tensors
            in_shape_.resize(3);
            in_size_ = 1;
            for (int i = 0; i < 3; ++i) {
                in_shape_[i] = X.dimension(i);
                in_size_ *= X.dimension(i);
            }

            if (start_dim_ == 1) {
                int batch_size = X.dimension(0);
                int feature_size = X.dimension(1) * X.dimension(2);
                out_size_ = feature_size;
                
                Eigen::array<int, 2> out_dims = {batch_size, feature_size};
                return X.reshape(out_dims);
            }
            
            // Default: flatten all dimensions
            Eigen::array<int, 2> out_dims = {1, static_cast<int>(X.size())};
            return X.reshape(out_dims);
        }

        Eigen::Tensor<double, 2> Flatten::forward(const Eigen::Tensor<double, 2>& X) {
            input_rank_ = 2;
            in_shape_.resize(2);
            for (int i = 0; i < 2; ++i) {
                in_shape_[i] = X.dimension(i);
            }
            in_size_ = X.size();
            out_size_ = X.size();
            return X; // Already 2D
        }

        Eigen::Tensor<double, 4> Flatten::backward4D(const Eigen::Tensor<double, 2>& dY) {
            if (input_rank_ != 4) {
                throw std::runtime_error("Flatten: backward4D called but input was not 4D");
            }
            if (in_shape_.size() != 4) {
                throw std::runtime_error("Flatten: Input shape not set for 4D tensor");
            }

            Eigen::array<int, 4> in_dims;
            for (int i = 0; i < 4; ++i) {
                in_dims[i] = in_shape_[i];
            }
            return dY.reshape(in_dims);
        }

        Eigen::Tensor<double, 3> Flatten::backward3D(const Eigen::Tensor<double, 2>& dY) {
            if (input_rank_ != 3) {
                throw std::runtime_error("Flatten: backward3D called but input was not 3D");
            }
            if (in_shape_.size() != 3) {
                throw std::runtime_error("Flatten: Input shape not set for 3D tensor");
            }

            Eigen::array<int, 3> in_dims;
            for (int i = 0; i < 3; ++i) {
                in_dims[i] = in_shape_[i];
            }
            return dY.reshape(in_dims);
        }

        Eigen::Tensor<double, 2> Flatten::backward2D(const Eigen::Tensor<double, 2>& dY) {
            return dY; // Already 2D
        }

        /************************************** Multi-Head Attention *******************************************/ 
        MultiHeadAttention::MultiHeadAttention
            (
                int in_size,
                int out_size,
                int num_heads = 8,
                int context_length = 512,
                double dropout_rate = 0.2,
                std::string layer_name = "Multi-Head Attention",
                bool trainable = true,
                bool qkv_bias = false


            ) :
            in_size_(in_size), out_size_(out_size),
            num_heads_(num_heads), context_length_(context_length),
            dropout_rate_(dropout_rate), layer_name_(layer_name),
            trainable_(trainable), qkv_bias_(qkv_bias)
            {
                head_size_ = in_size_ / num_heads_;
                bool is_divisible = in_size_ % num_heads_;
                if (!is_divisible)
                {
                    throw std::runtime_error("Multi-Head Attention: Input dimension must be divisible by the number of heads.");
                }   
            }
        Eigen::Tensor<double, 2> MultiHeadAttention::dense_forward(
            const Eigen::Tensor<double, 2>& X,
            Eigen::Tensor<double, 2>& W, 
            Eigen::Tensor<double, 1>& b)
        {
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> output = X.contract(W, product_dims);

            if (qkv_bias_) {
                
                Eigen::array<Eigen::Index, 2> broadcast_dims({X.dimension(0), 1});
                Eigen::Tensor<double, 2> bias_broadcasted = b.reshape(Eigen::array<Eigen::Index, 2>({1, b.dimension(0)})).broadcast(broadcast_dims);
                output = output + bias_broadcasted;
            }
            
            return output;
        }
        Eigen::Tensor<double, 2> MultiHeadAttention::dense_backward(
            const Eigen::Tensor<double, 2>& grad_out,
            Eigen::Tensor<double, 2>in_cache,
            Eigen::Tensor<double, 2> weights,
            Eigen::Tensor<double, 2>& grad_weights,
            Eigen::Tensor<double, 1>& grad_biases)
        {
            if (trainable_) {
                Eigen::array<int, 2> transpose_dims({1, 0});
                Eigen::Tensor<double, 2> X_transposed = in_cache.shuffle(transpose_dims);
                
                Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
                grad_weights = X_transposed.contract(grad_out, product_dims);
                
                if (qkv_bias_) 
                {
                    Eigen::array<int, 1> batch_dim({0});
                    grad_biases = grad_out.sum(batch_dim);
                }
            }
            Eigen::array<int, 2> transpose_dims({1, 0});
            Eigen::Tensor<double, 2> weights_transposed = weights.shuffle(transpose_dims);
            
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> grad_input = grad_out.contract(weights_transposed, product_dims);
            
            return grad_input;
        }

        void MultiHeadAttention::init_params_and_grads()
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            double scale = std::sqrt(6.0 / (in_size_ + out_size_));
            std::uniform_real_distribution<> dis(-scale, scale);

            // initialize weight-tensors
            Wq_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            Wk_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            Wv_ = Eigen::Tensor<double, 2>(in_size_, out_size_);

            for (int i = 0; i < in_size_; ++i) 
            {
                for (int j = 0; j < out_size_; ++j) 
                {
                    Wq_(i, j) = dis(gen);
                    Wk_(i, j) = dis(gen);
                    Wv_(i, j) = dis(gen);
                }
            }

            // initialize weight-gradient matrices with zero
            grad_Wq_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            grad_Wk_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            grad_Wv_ = Eigen::Tensor<double, 2>(in_size_, out_size_);

            grad_Wq_.setZero();
            grad_Wk_.setZero();
            grad_Wv_.setZero();

            if (qkv_bias_)
            {
                // initialize biases with zero, if bias_ is true.
                bq_ = Eigen::Tensor<double, 1>(out_size_);
                bk_ = Eigen::Tensor<double, 1>(out_size_);
                bv_ = Eigen::Tensor<double, 1>(out_size_);

                bq_.setZero();
                bk_.setZero();
                bv_.setZero();

                // initialize bias-gradient matrix with zero.
                grad_bq_ = Eigen::Tensor<double, 1>(out_size_);
                grad_bk_ = Eigen::Tensor<double, 1>(out_size_);
                grad_bv_ = Eigen::Tensor<double, 1>(out_size_);

                grad_bq_.setZero(); 
                grad_bk_.setZero();
                grad_bv_.setZero(); 
            }
            else
            {
                // initialize empty biases and gradients when bias is false
                bq_ = Eigen::Tensor<double, 1>(0);
                bk_ = Eigen::Tensor<double, 1>(0);
                bv_ = Eigen::Tensor<double, 1>(0);

                grad_bq_ = Eigen::Tensor<double, 1>(0);
                grad_bk_ = Eigen::Tensor<double, 1>(0);
                grad_bv_ = Eigen::Tensor<double, 1>(0);
            }
        }        
    }
}