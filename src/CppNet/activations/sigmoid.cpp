#include <cmath>
#include <Eigen/Dense>
#include "CppNet/activations/sigmoid.hpp"




namespace CppNet
{
    namespace Activations
    {
        /*****************************Sigmoid Activation Function *****************************/
        Sigmoid::Sigmoid(){}
        
        // helper method to set number of threads
        void Sigmoid::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }
        
        Eigen::Tensor<double, 2> Sigmoid::forward(const Eigen::Tensor<double, 2>& pre_activation) 
        {
            if (pre_activation.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Empty input tensor in 2D forward");
            }

            input_cache_2d_ = pre_activation; // cache input for backward pass
            int rows = pre_activation.dimension(0);
            int cols = pre_activation.dimension(1);
            output_cache_2d_ = Eigen::Tensor<double, 2>(rows, cols);
            output_cache_2d_.setZero();

            // compute sigmoid using element-wise operations and parallelization
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j) 
                {
                    // clamp input to prevent overflow/underflow
                    double clamped_input = std::max(-500.0, std::min(500.0, pre_activation(i, j)));
                    output_cache_2d_(i, j) = 1.0 / (1.0 + std::exp(-clamped_input));
                }
            }
            
            return output_cache_2d_;
        }

        Eigen::Tensor<double, 4> Sigmoid::forward(const Eigen::Tensor<double, 4>& pre_activation)
        {
            if (pre_activation.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Empty input tensor in 4D forward");
            }
            
            input_cache_4d_ = pre_activation; // cache input for backward pass
            
            int batch = pre_activation.dimension(0);
            int channels = pre_activation.dimension(1);
            int height = pre_activation.dimension(2);
            int width = pre_activation.dimension(3);
            
            output_cache_4d_ = Eigen::Tensor<double, 4>(batch, channels, height, width);
            output_cache_4d_.setZero();
            
            // compute sigmoid using element-wise operations and parallelization
            // collapse all 4 dimensions for maximum parallelization
            #pragma omp parallel for collapse(4) schedule(static)
            for (int b = 0; b < batch; ++b)
            {
                for (int c = 0; c < channels; ++c)
                {
                    for (int h = 0; h < height; ++h)
                    {
                        for (int w = 0; w < width; ++w)
                        {
                            // clamp input to prevent overflow/underflow
                            double clamped_input = std::max(-500.0, std::min(500.0, pre_activation(b, c, h, w)));
                            output_cache_4d_(b, c, h, w) = 1.0 / (1.0 + std::exp(-clamped_input));
                        }
                    }
                }
            }
            
            return output_cache_4d_;
        }

        Eigen::Tensor<double, 2> Sigmoid::backward(const Eigen::Tensor<double, 2>& grad_output)
        {  
            if (grad_output.dimension(0) != input_cache_2d_.dimension(0) ||
                grad_output.dimension(1) != input_cache_2d_.dimension(1) ||
                grad_output.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Shape mismatch or empty input in 2D backward");
            }

            int rows = grad_output.dimension(0);
            int cols = grad_output.dimension(1);
            Eigen::Tensor<double, 2> grad_input(rows, cols);
            grad_input.setZero();  

            // compute gradient using element-wise operations and parallelization
            #pragma omp parallel for collapse(2) schedule(static)
                for (int i = 0; i < rows; ++i) 
                {
                    for (int j = 0; j < cols; ++j) 
                    {
                        double sigmoid_val = output_cache_2d_(i, j);
                        grad_input(i, j) = grad_output(i, j) * sigmoid_val * (1.0 - sigmoid_val);
                    }
                }

            return grad_input;
        }

        Eigen::Tensor<double, 4> Sigmoid::backward(const Eigen::Tensor<double, 4>& grad_output)
        {
            if (grad_output.dimension(0) != input_cache_4d_.dimension(0) ||
                grad_output.dimension(1) != input_cache_4d_.dimension(1) ||
                grad_output.dimension(2) != input_cache_4d_.dimension(2) ||
                grad_output.dimension(3) != input_cache_4d_.dimension(3) ||
                grad_output.size() == 0)
            {
                throw std::runtime_error("Sigmoid: Shape mismatch or empty input in 4D backward");
            }
            
            int batch = grad_output.dimension(0);
            int channels = grad_output.dimension(1);
            int height = grad_output.dimension(2);
            int width = grad_output.dimension(3);
            
            Eigen::Tensor<double, 4> grad_input(batch, channels, height, width);
            grad_input.setZero();
            
            // compute gradient using element-wise operations and parallelization
            // collapse all 4 dimensions for maximum parallelization
            #pragma omp parallel for collapse(4) schedule(static)
            for (int b = 0; b < batch; ++b)
            {
                for (int c = 0; c < channels; ++c)
                {
                    for (int h = 0; h < height; ++h)
                    {
                        for (int w = 0; w < width; ++w)
                        {
                            double sigmoid_val = output_cache_4d_(b, c, h, w);
                            grad_input(b, c, h, w) = grad_output(b, c, h, w) * sigmoid_val * (1.0 - sigmoid_val);
                        }
                    }
                }
            }
            
            return grad_input;
        }
    }
}