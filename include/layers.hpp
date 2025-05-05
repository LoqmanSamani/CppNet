#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <Eigen/Dense>
#include <random>

namespace CppNet {
    class Optimizer; // Forward declaration

    class Layer {
    public:
        virtual bool is_trainable() const = 0;
        virtual void update_parameters(Optimizer& optimizer, double learning_rate) = 0;
        virtual ~Layer() = default;
    };

    class Linear : public Layer
    {
    public:
        bool trainable_; // if gradient should be calculated. if false: layer is frozen.

        Linear(int in_size, int out_size, std::string layer_name = "Linear", bool trainable = true, bool bias = true);

        Eigen::MatrixXd forward(const Eigen::MatrixXd& X);

        Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_out);

        void reset_gradients() {
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

        void init_params_and_grads();
    };
}

#endif // LAYERS_HPP