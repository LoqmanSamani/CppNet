#ifndef LAYERS_HPP
#define LAYERS_HPP

#include<Eigen/Dense>
#include<random>


namespace CppNet
{
    class Linear
    {
        public:

            Linear(int in_size, int out_size);

            Eigen::MatrixXd forward(const Eigen::MatrixXd& X);

            Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_out);

            void update_params(double lr);

            Eigen::MatrixXd get_weights() const {return weights_;}
            Eigen::VectorXd get_biases() const { return biases_;}


        private:

            int in_size_;
            int out_size_;
            Eigen::MatrixXd weights_;
            Eigen::VectorXd biases_;
            Eigen::MatrixXd in_cache_;
            Eigen::MatrixXd grad_weights_;
            Eigen::VectorXd grad_biases_;

            void init_params();
        
    };
}



#endif // LAYERS_HPP
