#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <tuple>

namespace CppNet 
{
    class Optimizer; // forward declaration

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

            Eigen::MatrixXd forward(const Eigen::MatrixXd& X);

            Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_out);

            void reset_grads() {
                grad_weights_.setZero();
                if (bias_) {
                    grad_biases_.setZero();
                }
            }

            Eigen::MatrixXd& get_weights() { return weights_; }
            const Eigen::MatrixXd& get_weights() const { return weights_; }
            Eigen::VectorXd& get_biases() { return biases_; }
            const Eigen::VectorXd& get_biases() const { return biases_; }
            const Eigen::MatrixXd& get_grad_weights() const { return grad_weights_; }
            const Eigen::VectorXd& get_grad_biases() const { return grad_biases_; }
            void set_weights(const Eigen::MatrixXd& weights) { weights_ = weights; }
            void set_biases(const Eigen::VectorXd& biases) { biases_ = biases; }
            std::string get_layer_name() const { return layer_name_; }
            bool is_trainable() const override { return trainable_; }
            void freeze(){ trainable_ = false; } // freeze the parameters of the layer.
            void unfreeze() { trainable_ = true; } // unfreeze the parameters of the layer.
            bool has_bias() const { return bias_; }
            void update_parameters(Optimizer& optimizer, double learning_rate) override;

        private:
            int in_size_;
            int out_size_;
            bool bias_;
            std::string layer_name_;
            Eigen::MatrixXd weights_;
            Eigen::VectorXd biases_;
            Eigen::MatrixXd in_cache_;
            Eigen::MatrixXd grad_weights_;
            Eigen::VectorXd grad_biases_;
            bool trainable_; // if gradient should be calculated. if false: layer is frozen.

            void init_params_and_grads();
    };

    class Conv2d: public Layer
    {
        public:

            Conv2d(
                int in_channels,
                int out_channels,
                std::tuple<int, int> kernel_size = std::make_tuple(3, 3),
                std::tuple<int, int> stride = std::make_tuple(1, 1), 
                std::string padding = "valid", 
                std::tuple<int, int> num_padding = std::make_tuple(2, 2), 
                std::string padding_mode = "zero", 
                std::string layer_name = "Conv2D", 
                bool trainable = true, 
                bool bias = true
            );

            Eigen::Tensor<double, 4> forward(Eigen::Tensor<double, 4>& X);
            Eigen::Tensor<double, 4> backward(Eigen::Tensor<double, 4>& grad_out);

            void reset_grads() {
                grad_weights_.setZero();
                if (bias_) {
                    grad_biases_.setZero();
                }
            }

            Eigen::Tensor<double, 4>& get_weights() { return weights_; }
            const Eigen::Tensor<double, 4>& get_weights() const { return weights_; }

            //Eigen::Tensor<double, 3>& get_weights() { return weights_; }
            //const Eigen::Tensor<double, 3>& get_weights() const { return weights_; }

            Eigen::VectorXd& get_biases() { return biases_; }
            const Eigen::VectorXd& get_biases() const { return biases_; }
            const Eigen::Tensor<double, 4>& get_grad_weights() const { return grad_weights_; }
            //const Eigen::Tensor<double, 3>& get_grad_weights() const { return grad_weights_; }
            const Eigen::VectorXd& get_grad_biases() const { return grad_biases_; }
            void set_weights(const Eigen::Tensor<double, 4>& weights) { weights_ = weights; }
            //void set_weights(const Eigen::Tensor<double, 3>& weights) { weights_ = weights; }
            void set_biases(const Eigen::VectorXd& biases) { biases_ = biases; }
            std::string get_layer_name() const { return layer_name_; }
            bool is_trainable() const override { return trainable_; }
            void freeze(){ trainable_ = false; } // freeze the parameters of the layer.
            void unfreeze() { trainable_ = true; } // unfreeze the parameters of the layer.
            bool has_bias() const { return bias_; }
            void update_parameters(Optimizer& optimizer, double learning_rate) override;

        private:

            int in_channels_;
            int out_channels_;
            bool trainable_;
            std::tuple<int, int> stride_;
            std::string padding_;
            std::tuple<int, int, int, int> num_padding_; // if padding is set to "none", the amount of padding from this tuple will be used to padding (left, right, top bottom)
            std::string padding_mode_;
            bool bias_;
            Eigen::Tensor<double, 4> in_cache_;
            std::tuple<int, int> kernel_size_;
            std::string layer_name_;
            Eigen::Tensor<double, 4> weights_;
            //Eigen::Tensor<double, 3> weights_;
            Eigen::VectorXd biases_;
            Eigen::Tensor<double, 4> in_cache_;
            Eigen::Tensor<double, 4> output_;
            Eigen::Tensor<double, 4> grad_weights_;
            //Eigen::Tensor<double, 3> grad_weights_;
            Eigen::VectorXd grad_biases_;

            // input shape
            int B_; // batch size
            int C_; // number of channels
            int H_; // heigth of data (e.g., images)
            int W_; // width of data (e.g., images)
            int h_; // heigth of output data
            int w_;
            
            void init_params_and_grads();
            void init_output();
            Eigen::Tensor<double, 4> pad_input(Eigen::Tensor<double, 4>& X);
    };
}

#endif // LAYERS_HPP