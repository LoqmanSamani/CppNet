#include <cmath>
#include <Eigen/Dense>
#include "CppNet/activations/relu.hpp"




namespace CppNet
{
    namespace Activations
    {
        /************************* ReLU Activation Function **************************/
        ReLU::ReLU(){}

        void ReLU::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }

        Eigen::Tensor<float, 2> ReLU::forward(const Eigen::Tensor<float, 2>& pre_activation) 
        {
            if (pre_activation.size() == 0)
            {
                throw std::runtime_error("ReLU: Empty input tensor in 2D forward");
            }
            int rows = pre_activation.dimension(0);
            int cols = pre_activation.dimension(1);
            output_cache_2d_ = Eigen::Tensor<float, 2>(rows, cols);
            output_cache_2d_.setZero(); 

            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j) 
                {
                    output_cache_2d_(i, j) = std::max(0.0f, pre_activation(i, j));
                }
            }
            return output_cache_2d_;

        }

        Eigen::Tensor<float, 4> ReLU::forward(const Eigen::Tensor<float, 4>& pre_activation)
        {
            if (pre_activation.size() == 0)
            {
                throw std::runtime_error("ReLU: Empty input tensor in 4D forward");
            }
            
            int batch = pre_activation.dimension(0);
            int channels = pre_activation.dimension(1);
            int height = pre_activation.dimension(2);
            int width = pre_activation.dimension(3);
            
            output_cache_4d_ = Eigen::Tensor<float, 4>(batch, channels, height, width);
            output_cache_4d_.setZero();
            
            // Compute ReLU using element-wise operations and parallelization
            // Collapse all 4 dimensions for maximum parallelization
            #pragma omp parallel for collapse(4) schedule(static)
            for (int b = 0; b < batch; ++b)
            {
                for (int c = 0; c < channels; ++c)
                {
                    for (int h = 0; h < height; ++h)
                    {
                        for (int w = 0; w < width; ++w)
                        {
                            output_cache_4d_(b, c, h, w) = std::max(0.0f, pre_activation(b, c, h, w));
                        }
                    }
                }
            }
            
            return output_cache_4d_;
        }

        Eigen::Tensor<float, 2> ReLU::backward(const Eigen::Tensor<float, 2>& grad_output) 
        {
            if (grad_output.dimension(0) != output_cache_2d_.dimension(0) ||
                grad_output.dimension(1) != output_cache_2d_.dimension(1) ||
                grad_output.size() == 0)
            {
                throw std::runtime_error("ReLU: Shape mismatch or empty input in 2D backward");
            }

            int rows = grad_output.dimension(0);
            int cols = grad_output.dimension(1);
            Eigen::Tensor<float, 2> grad_input(rows, cols);
            grad_input.setZero();

            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j) {
                    grad_input(i, j) = grad_output(i, j) * (output_cache_2d_(i, j) > 0.0 ? 1.0 : 0.0);
                }
            }
            
            return grad_input;
        }

        Eigen::Tensor<float, 4> ReLU::backward(const Eigen::Tensor<float, 4>& grad_output)
        {
            if (grad_output.dimension(0) != output_cache_4d_.dimension(0) ||
                grad_output.dimension(1) != output_cache_4d_.dimension(1) ||
                grad_output.dimension(2) != output_cache_4d_.dimension(2) ||
                grad_output.dimension(3) != output_cache_4d_.dimension(3) ||
                grad_output.size() == 0)
            {
                throw std::runtime_error("ReLU: Shape mismatch or empty input in 4D backward");
            }
            
            int batch = grad_output.dimension(0);
            int channels = grad_output.dimension(1);
            int height = grad_output.dimension(2);
            int width = grad_output.dimension(3);
            
            Eigen::Tensor<float, 4> grad_input(batch, channels, height, width);
            grad_input.setZero();
            
            #pragma omp parallel for collapse(4) schedule(static)
            for (int b = 0; b < batch; ++b)
            {
                for (int c = 0; c < channels; ++c)
                {
                    for (int h = 0; h < height; ++h)
                    {
                        for (int w = 0; w < width; ++w)
                        {
                            grad_input(b, c, h, w) = grad_output(b, c, h, w) * (output_cache_4d_(b, c, h, w) > 0.0 ? 1.0 : 0.0);
                        }
                    }
                }
            }
            
            return grad_input;
        }
    }
}