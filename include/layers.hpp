#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>

namespace CppNet
{
    namespace Optimizers
    {
        class Optimizer;
    }
    //namespace Activations
    //{
    //    class Activation;
    //    class ReLU;
    //}
    
    namespace Layers
    {
        
        class Layer 
        {
            public:
                virtual bool is_trainable() const = 0;
                virtual void step(Optimizers::Optimizer& optimizer, double learning_rate) = 0;
                virtual ~Layer() = default;
        };

        //********************* Linear (Fully Connected: Dense) Layer *********************//
        class Linear : public Layer
        {
            public:

                Linear(
                    int in_size, 
                    int out_size, 
                    std::string layer_name = "Linear", 
                    bool trainable = true, 
                    bool bias = true,
                    std::string device = "cpu",
                    std::string weight_init = "xavier",
                    int parallel_threshold = 10000
                ); 
                
                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& input);
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& grad_output);
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

                void step(Optimizers::Optimizer& optimizer, double learning_rate) override;

                // OpenMP utility method
                static void set_num_threads(int num_threads);

                
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
          
                int in_size_;
                int out_size_;
                std::string layer_name_;
                bool trainable_; // if gradient should be calculated. if false: layer is frozen.
                bool bias_;
                std::string device_;
                std::string weight_init_;
                int parallel_threshold_; // threshold for input size to enable parallelization    
                Eigen::Tensor<double, 2> weights_;
                Eigen::Tensor<double, 1> biases_;
                Eigen::Tensor<double, 2> in_cache_;
                Eigen::Tensor<double, 2> grad_weights_;
                Eigen::Tensor<double, 1> grad_biases_;

                void reinitialize_weights(const std::string& new_init_method);
                void init_params_and_grads();
                         
        };

        //********************* Convolutional Layer(2D) *********************//
        class Conv2d : public Layer
        {
            public:
                
                Conv2d(
                    int in_chanlels,
                    int out_channels,
                    std::tuple<int , int> kernel_size = std::make_tuple(3, 3),
                    std::tuple<int, int> stride = std::make_tuple(1, 1),
                    std::string layer_name = "Conv2D",
                    bool trainable = true,
                    bool bias = true,
                    std::string padding = "valid",
                    std::tuple<int, int, int, int> num_padding = std::make_tuple(0, 0, 0, 0),
                    std::string padding_mode = "zero",  
                    std::string device = "cpu",
                    std::string weight_init = "xavier",
                    int parallel_threshold = 10000
                );

                Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& input);
                Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& grad_output);

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

                void step(Optimizers::Optimizer& optimizer, double learning_rate) override;

                // OpenMP utility method
                static void set_num_threads(int num_threads);
                
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
                std::tuple<int, int> kernel_size_;
                std::tuple<int, int> stride_;
                std::string layer_name_;
                bool trainable_; // if gradient should be calculated. if false: layer is frozen.
                bool bias_;
                std::string padding_;
                std::tuple<int, int, int, int> num_padding_;
                std::string padding_mode_;
                std::string device_;
                std::string weight_init_;
                int parallel_threshold_; // threshold for input size to enable parallelization  
                // Input/output dimensions
                int B_, C_, H_, W_; // Input dimensions
                int h_, w_;         // Output dimensions

                //std::unique_ptr<Activations::ReLU> default_relu_;
                //Activations::Activation* activation_;
                Eigen::Tensor<double, 4> weights_;
                Eigen::Tensor<double, 1> biases_;
                Eigen::Tensor<double, 4> grad_weights_;
                Eigen::Tensor<double, 1> grad_biases_;
                Eigen::Tensor<double, 4> in_cache_;
                Eigen::Tensor<double, 4> output_;

                void reinitialize_weights(const std::string& new_init_method);
                void init_params_and_grads();
                void init_output();
                Eigen::Tensor<double, 4> pad_input(const Eigen::Tensor<double, 4>& input);
                Eigen::Tensor<double, 2> im2col(const Eigen::Tensor<double, 4>& input);
                void col2im_add(const Eigen::Tensor<double, 2>& grad_input_col, Eigen::Tensor<double, 4>& grad_input);
                Eigen::Tensor<double, 4> apply_manual_padding(const Eigen::Tensor<double, 4>& input, int pad_top, int pad_bottom, int pad_left, int pad_right, std::string padding_type);
                void apply_horizontal_padding(Eigen::Tensor<double, 4>& padded_input, int batch_size, int channels, int padded_height, int width, int pad_left, int pad_right, std::string padding_type, bool use_parallel);
                void apply_vertical_padding(Eigen::Tensor<double, 4>& padded_input, int batch_size, int channels, int height, int pad_top, int pad_bottom, std::string padding_type, bool use_parallel);
                
        };

        //********************* Max Pool 2D *********************//
        class MaxPool2D : public Layer
        {
            public:

                MaxPool2D(
                        std::tuple<int, int> kernel_size = std::make_tuple(3, 3),
                        std::tuple<int, int> stride = std::make_tuple(1, 1),
                        std::string layer_name = "MaxPool2D",
                        std::string padding = "valid",
                        std::tuple<int, int, int, int> num_padding = std::make_tuple(0, 0, 0, 0),
                        std::string padding_mode = "zero",
                        std::string device = "cpu",
                        int parallel_threshold = 10000   
                    );  
                Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& input);
                Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& grad_output);

                void reset_grads() {} // MaxPool has no gradients to reset
                bool is_trainable() const override { return false; } // MaxPool is not trainable
                void step(Optimizers::Optimizer& optimizer, double learning_rate) override {}
                std::string get_layer_name() const { return layer_name_; }

                // OpenMP utility method
                static void set_num_threads(int num_threads);
                
                void print_layer_info() const {
                    std::cout << "Layer: " << layer_name_ << std::endl;
                    std::cout << " Kernel size: [" << std::get<0>(kernel_size_) << ", " << std::get<1>(kernel_size_) << "]" << std::endl;
                    std::cout << " Stride: [" << std::get<0>(stride_) << ", " << std::get<1>(stride_) << "]" << std::endl;
                    std::cout << " Padding: " << padding_ << std::endl;
                }

            private:

                std::tuple<int, int> kernel_size_;
                std::tuple<int, int> stride_;
                std::string layer_name_;
                std::string padding_;
                std::tuple<int, int, int, int> num_padding_;
                std::string padding_mode_;
                std::string device_;
                int parallel_threshold_; // threshold for input size to enable parallelization
                bool trainable_ = false; // MaxPool is not trainable   
                Eigen::Tensor<double, 4> in_cache_;
                Eigen::Tensor<double, 4> output_;
                int B_, C_, H_, W_; // input dimensions
                int h_, w_; // output dimensions
                
                void init_output();
                Eigen::Tensor<double, 4> pad_input(const Eigen::Tensor<double, 4>& input);
                Eigen::Tensor<double, 4> apply_manual_padding(const Eigen::Tensor<double, 4>& input, int pad_top, int pad_bottom, int pad_left, int pad_right, std::string padding_type);
                void apply_horizontal_padding(Eigen::Tensor<double, 4>& padded_input, int batch_size, int channels, int padded_height, int width, int pad_left, int pad_right, std::string padding_type, bool use_parallel);
                void apply_vertical_padding(Eigen::Tensor<double, 4>& padded_input, int batch_size, int channels, int height, int pad_top, int pad_bottom, std::string padding_type, bool use_parallel);
                
        };
        
        //********************* Flatten Layer *********************//
        class Flatten : public Layer
        {
            public:
                Flatten(std::string layer_name = "Flatten");

                Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 4>& input); 
                Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 2>& grad_output);

                Eigen::Tensor<double, 2> forward_3d(const Eigen::Tensor<double, 3>& input); 
                Eigen::Tensor<double, 3> backward_3d(const Eigen::Tensor<double, 2>& grad_output);

                bool is_trainable() const override { return false; }
                void step(Optimizers::Optimizer& optimizer, double learning_rate) override {}
                std::string get_layer_name() const { return layer_name_; }

                // OpenMP utility method
                static void set_num_threads(int num_threads);
                
            private:

                std::string layer_name_;
                std::array<Eigen::Index, 2> in_shape_2d;
                std::array<Eigen::Index, 3> in_shape_3d;
                std::array<Eigen::Index, 4> in_shape_4d;
                Eigen::Tensor<double, 4> grad_input_;
                int B_, C_, H_, W_; // input dimensions
                      
        };

        //********************* Multi-Head Attention *********************//
        class MultiHeadAttention : public Layer
        {
            public:
          
                MultiHeadAttention
                (
                    int in_size,
                    int out_size,
                    int num_heads = 8,
                    int context_length = 512,
                    double dropout_rate = 0.1,
                    bool trainable = true,
                    bool qkv_bias = false,
                    std::string layer_name = "Multi-Head Attention",
                    std::string device = "cpu",
                    std::string weight_init = "xavier",
                    int parallel_threshold = 10000
                );

                Eigen::Tensor<double, 3> forward(Eigen::Tensor<double, 3>& inputs, Eigen::Tensor<double, 3>& targets = empty_tensor, bool apply_mask = false);
                Eigen::Tensor<double, 3> backward(Eigen::Tensor<double, 3>&grad_outputs, Eigen::Tensor<double, 3>& grad_targets = empty_tensor);


                void reset_grads() 
                {
                    if (trainable_) 
                    {
                        grad_Wq_.setZero();
                        grad_Wk_.setZero();
                        grad_Wv_.setZero();

                        if (qkv_bias_ && grad_bq_.size() > 0 && grad_bk_.size() > 0 && grad_bv_.size() > 0) 
                        { 
                            grad_bq_.setZero();
                            grad_bk_.setZero();
                            grad_bv_.setZero();
                        }
                    } 
                }

                int get_input_size() const { return in_size_; }
                int get_output_size() const { return out_size_; }
                std::string get_layer_name() const { return layer_name_; }
                int get_num_heads() const { return num_heads_; }
                int get_context_length() const { return context_length_; }
                int get_dropout_rate() const{ return dropout_rate_; }

                Eigen::Tensor<double, 2>& get_query_weights() { return Wq_; }
                const Eigen::Tensor<double, 2>& get_query_weights() const { return Wq_; }
                Eigen::Tensor<double, 2>& get_key_weights() { return Wk_; }
                const Eigen::Tensor<double, 2>& get_key_weights() const { return Wk_; }
                Eigen::Tensor<double, 2>& get_value_weights() { return Wv_; }
                const Eigen::Tensor<double, 2>& get_value_weights() const { return Wv_; }

                Eigen::Tensor<double, 1>& get_query_biases() { return bq_; }
                const Eigen::Tensor<double, 1>& get_query_biases() const { return bq_; }
                Eigen::Tensor<double, 1>& get_key_biases() { return bk_; }
                const Eigen::Tensor<double, 1>& get_key_biases() const { return bk_; }
                Eigen::Tensor<double, 1>& get_value_biases() { return bv_; }
                const Eigen::Tensor<double, 1>& get_value_biases() const { return bv_; }

                const Eigen::Tensor<double, 2>& get_grad_query_weights() const { return grad_Wq_; }
                const Eigen::Tensor<double, 1>& get_grad_query_biases() const { return grad_bq_; }
                const Eigen::Tensor<double, 2>& get_grad_key_weights() const { return grad_Wk_; }
                const Eigen::Tensor<double, 1>& get_grad_key_biases() const { return grad_bk_; }
                const Eigen::Tensor<double, 2>& get_grad_value_weights() const { return grad_Wv_; }
                const Eigen::Tensor<double, 1>& get_grad_value_biases() const { return grad_bv_; }

                bool is_trainable() const override { return trainable_; }

                void freeze(){ trainable_ = false; }
                void unfreeze() { trainable_ = true; }

                bool has_bias() const { return qkv_bias_; }

                void step(Optimizers::Optimizer& optimizer, double learning_rate) override;
                
                void print_layer_info() const 
                {
                    std::cout << "  Layer: " << layer_name_ << std::endl;
                    std::cout << "  Input Dimension: " << in_size_ << std::endl;
                    std::cout << "  Output Dimension: " << out_size_ << std::endl;
                    std::cout << "  Trainable: " << (trainable_ ? "Yes" : "No") << std::endl;
                    std::cout << "  Has Bias: " << (qkv_bias_ ? "Yes" : "No") << std::endl;
                    std::cout << "  Query Weight Shape: [" << Wq_.dimension(0) << ", " << Wq_.dimension(1) << "]" << std::endl;
                    std::cout << "  Key Weight Shape: [" << Wk_.dimension(0) << ", " << Wk_.dimension(1) << "]" << std::endl;
                    std::cout << "  Value Weight Shape: [" << Wv_.dimension(0) << ", " << Wv_.dimension(1) << "]" << std::endl;

                    if (qkv_bias_)
                    {
                        std::cout << "  Query Bias shape: [" << bq_.dimension(0) << "]" << std::endl;
                        std::cout << "  Key Bias shape: [" << bk_.dimension(0) << "]" << std::endl;
                        std::cout << "  Value Bias shape: [" << bv_.dimension(0) << "]" << std::endl;
                    }
                }

            private:
           
                //bool trainable_;
                int in_size_;
                int out_size_;
                int num_heads_;
                int context_length_;
                double dropout_rate_;
                bool trainable_;
                bool qkv_bias_;
                std::string layer_name_;
                std::string device_;
                std::string weight_init_;
                int parallel_threshold_; // threshold for input size to enable parallelization
                int head_size_;
                Eigen::Tensor<bool, 2> mask_;
                Flatten flatten_;
                Eigen::Tensor<double, 2> softmax_outputs_;
                
                //Flatten flatten_(std::string layer_name = "Flatten_for_MHA");
                
                // weight matrices
                Eigen::Tensor<double, 2> Wq_;
                Eigen::Tensor<double, 2> Wk_;
                Eigen::Tensor<double, 2> Wv_;
                Eigen::Tensor<double, 1> bq_;
                Eigen::Tensor<double, 1> bk_;
                Eigen::Tensor<double, 1> bv_;
                
                // gradient matrices
                Eigen::Tensor<double, 2> grad_Wq_;
                Eigen::Tensor<double, 2> grad_Wk_;
                Eigen::Tensor<double, 2> grad_Wv_;
                Eigen::Tensor<double, 1> grad_bq_;
                Eigen::Tensor<double, 1> grad_bk_;
                Eigen::Tensor<double, 1> grad_bv_;
                
                // cache variables for backward pass
                Eigen::Tensor<double, 2> in_cache_;      // for self-attention: flattened inputs
                Eigen::Tensor<double, 2> X_cache_;       // for cross-attention: flattened targets
                Eigen::Tensor<double, 2> Y_cache_;       // for cross-attention: flattened targets
                Eigen::Tensor<double, 4> Q_cache_;       // cached Query tensor
                Eigen::Tensor<double, 4> K_cache_;       // cached Key tensor
                Eigen::Tensor<double, 4> V_cache_;       // cached Value tensor
                Eigen::Tensor<double, 4> attention_weights_cache_; // cached attention weights
                
                static Eigen::Tensor<double, 3> empty_tensor;
                
                void init_params_and_grads();
                void reinitialize_weights(const std::string& new_init_method);
                Eigen::Tensor<double, 2> dense_forward(const Eigen::Tensor<double, 2>& inputs, Eigen::Tensor<double, 2>& weights, Eigen::Tensor<double, 1>& biases);
                Eigen::Tensor<double, 2> dense_backward(const Eigen::Tensor<double, 2>& grad_outputs, Eigen::Tensor<double, 2>in_cache, Eigen::Tensor<double, 2> weights, Eigen::Tensor<double, 2>& grad_weights, Eigen::Tensor<double, 1>& grad_biases);
                Eigen::Tensor<bool, 2> create_causal_mask(int context_length);
                Eigen::Tensor<double, 2> softmax_forward(const Eigen::Tensor<double, 2>& inputs);
                Eigen::Tensor<double, 2> softmax_backward(const Eigen::Tensor<double, 2>& grad_outputs);
                void apply_causal_mask(Eigen::Tensor<double, 4>& attention_scores, const Eigen::Tensor<bool, 2>& mask, int num_tokens);
        };

        class RNN : public Layer
        {
            public:
                RNN();
                Eigen::Tensor<double, 3> forward(const Eigen::Tensor<double, 3>& input); 
                Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& grad_output);
        };
    }
}

#endif // LAYERS_HPP