#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "layers.hpp"
#include "optimizers.hpp"




namespace CppNet
{
    namespace Activations
    {
        // base activation class
        class Activation : public Layers::Layer
        {
        public:

            virtual Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z) = 0;
            virtual Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da) = 0;
            
            virtual double forward(double z) = 0;
            virtual Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& z) = 0;
            virtual Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& da) = 0;
            
            bool is_trainable() const override final { return false; }
            void update_parameters(Optimizers::Optimizer&, double) override final {}
            virtual ~Activation() = default;
        };

        class ReLU : public Activation
        {
        public:
            
            Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z) override;
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da) override;
            
            double forward(double z) override;
            
            Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& z) override;
            Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& da) override;

        private:
            Eigen::Tensor<double, 2> in_cache_2d_;
            Eigen::Tensor<double, 4> in_cache_4d_;
        };

        class Sigmoid : public Activation
        {
        public:
        
            Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2>& z) override;
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& da) override;
            
            double forward(double z) override;
        
            Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& z) override;
            Eigen::Tensor<double, 4> backward(const Eigen::Tensor<double, 4>& da) override;

        private:

            Eigen::Tensor<double, 2> in_cache_2d_;
            Eigen::Tensor<double, 2> sigmoid_cache_2d_;
            Eigen::Tensor<double, 4> in_cache_4d_;
            Eigen::Tensor<double, 4> sigmoid_cache_4d_;
        };
    }
}




#endif // ACTIVATIONS_HPP