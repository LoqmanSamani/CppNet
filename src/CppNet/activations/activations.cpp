#include <cmath>
#include <Eigen/Dense>
#include "activations.hpp"

namespace CppNet
{
    namespace Activations
    {
        /************************************** Sigmoid *************************************/
        Sigmoid::Sigmoid() 
        {
            // No parameters needed for Sigmoid
        }
        
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

            input_cache_2d_ = pre_activation; // Cache input for backward pass
            int rows = pre_activation.dimension(0);
            int cols = pre_activation.dimension(1);
            output_cache_2d_ = Eigen::Tensor<double, 2>(rows, cols);
            output_cache_2d_.setZero();

            // Compute sigmoid using element-wise operations and parallelization
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j) 
                {
                    // Clamp input to prevent overflow/underflow
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
            
            input_cache_4d_ = pre_activation; // Cache input for backward pass
            
            int batch = pre_activation.dimension(0);
            int channels = pre_activation.dimension(1);
            int height = pre_activation.dimension(2);
            int width = pre_activation.dimension(3);
            
            output_cache_4d_ = Eigen::Tensor<double, 4>(batch, channels, height, width);
            output_cache_4d_.setZero();
            
            // Compute sigmoid using element-wise operations and parallelization
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
                            // Clamp input to prevent overflow/underflow
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

            // Compute gradient using element-wise operations and parallelization
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
            
            // Compute gradient using element-wise operations and parallelization
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
                            double sigmoid_val = output_cache_4d_(b, c, h, w);
                            grad_input(b, c, h, w) = grad_output(b, c, h, w) * sigmoid_val * (1.0 - sigmoid_val);
                        }
                    }
                }
            }
            
            return grad_input;
        }
      
       
        /************************************** ReLU *************************************/
        ReLU::ReLU() 
        {
            // No parameters needed for ReLU
        }
        // helper method to set number of threads
        void ReLU::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }

        Eigen::Tensor<double, 2> ReLU::forward(const Eigen::Tensor<double, 2>& pre_activation) 
        {
            if (pre_activation.size() == 0)
            {
                throw std::runtime_error("ReLU: Empty input tensor in 2D forward");
            }
            int rows = pre_activation.dimension(0);
            int cols = pre_activation.dimension(1);
            output_cache_2d_ = Eigen::Tensor<double, 2>(rows, cols);
            output_cache_2d_.setZero(); 

            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j) {
                    output_cache_2d_(i, j) = std::max(0.0, pre_activation(i, j));
                }
            }
            return output_cache_2d_;

        }

        Eigen::Tensor<double, 4> ReLU::forward(const Eigen::Tensor<double, 4>& pre_activation)
        {
            if (pre_activation.size() == 0)
            {
                throw std::runtime_error("ReLU: Empty input tensor in 4D forward");
            }
            
            int batch = pre_activation.dimension(0);
            int channels = pre_activation.dimension(1);
            int height = pre_activation.dimension(2);
            int width = pre_activation.dimension(3);
            
            output_cache_4d_ = Eigen::Tensor<double, 4>(batch, channels, height, width);
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
                            output_cache_4d_(b, c, h, w) = std::max(0.0, pre_activation(b, c, h, w));
                        }
                    }
                }
            }
            
            return output_cache_4d_;
        }

        Eigen::Tensor<double, 2> ReLU::backward(const Eigen::Tensor<double, 2>& grad_output) 
        {
            if (grad_output.dimension(0) != output_cache_2d_.dimension(0) ||
                grad_output.dimension(1) != output_cache_2d_.dimension(1) ||
                grad_output.size() == 0)
            {
                throw std::runtime_error("ReLU: Shape mismatch or empty input in 2D backward");
            }

            int rows = grad_output.dimension(0);
            int cols = grad_output.dimension(1);
            Eigen::Tensor<double, 2> grad_input(rows, cols);
            grad_input.setZero();

            // Element-wise multiplication with the derivative mask and parallelization
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j) {
                    grad_input(i, j) = grad_output(i, j) * (output_cache_2d_(i, j) > 0.0 ? 1.0 : 0.0);
                }
            }
            
            return grad_input;
        }

        Eigen::Tensor<double, 4> ReLU::backward(const Eigen::Tensor<double, 4>& grad_output)
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
            
            Eigen::Tensor<double, 4> grad_input(batch, channels, height, width);
            grad_input.setZero();
            
            // Element-wise multiplication with the derivative mask and parallelization
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
                            grad_input(b, c, h, w) = grad_output(b, c, h, w) * (output_cache_4d_(b, c, h, w) > 0.0 ? 1.0 : 0.0);
                        }
                    }
                }
            }
            
            return grad_input;
        }

        //Eigen::Tensor<double, 3> ReLU::forward(const Eigen::Tensor<double, 3>& input) 
        //{
        //    Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
        //    output.setZero();
        //    return output;
        //}

        //Eigen::Tensor<double, 3> ReLU::backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input) 
        //{
        //    Eigen::Tensor<double, 3> grad_input(input.dimension(0), input.dimension(1), input.dimension(2));
        //    grad_input.setZero();
        //    return grad_input;
        //}

        /************************************** Tanh *************************************/
        Tanh::Tanh() 
        {
            // No parameters needed for Tanh
        }

        Eigen::Tensor<double, 1> Tanh::forward(const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            Eigen::Tensor<double, 1> output(input.dimension(0));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 1> Tanh::backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement tanh derivative: 1 - tanh²(x)
            Eigen::Tensor<double, 1> grad_input(input.dimension(0));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 2> Tanh::forward(const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> output(input.dimension(0), input.dimension(1));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 2> Tanh::backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> grad_input(input.dimension(0), input.dimension(1));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 3> Tanh::forward(const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 3> Tanh::backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> grad_input(input.dimension(0), input.dimension(1), input.dimension(2));
            grad_input.setZero();
            return grad_input;
        }

        

        /************************************** LeakyReLU *************************************/
        LeakyReLU::LeakyReLU(double negative_slope) : negative_slope(negative_slope) 
        {
            // Store the negative slope parameter
        }

        Eigen::Tensor<double, 1> LeakyReLU::forward(const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement LeakyReLU: x if x > 0, else negative_slope * x
            Eigen::Tensor<double, 1> output(input.dimension(0));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 1> LeakyReLU::backward(const Eigen::Tensor<double, 1>& grad_output, const Eigen::Tensor<double, 1>& input) 
        {
            // TODO: Implement LeakyReLU derivative: 1 if x > 0, else negative_slope
            Eigen::Tensor<double, 1> grad_input(input.dimension(0));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 2> LeakyReLU::forward(const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> output(input.dimension(0), input.dimension(1));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 2> LeakyReLU::backward(const Eigen::Tensor<double, 2>& grad_output, const Eigen::Tensor<double, 2>& input) 
        {
            Eigen::Tensor<double, 2> grad_input(input.dimension(0), input.dimension(1));
            grad_input.setZero();
            return grad_input;
        }

        Eigen::Tensor<double, 3> LeakyReLU::forward(const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> output(input.dimension(0), input.dimension(1), input.dimension(2));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 3> LeakyReLU::backward(const Eigen::Tensor<double, 3>& grad_output, const Eigen::Tensor<double, 3>& input) 
        {
            Eigen::Tensor<double, 3> grad_input(input.dimension(0), input.dimension(1), input.dimension(2));
            grad_input.setZero();
            return grad_input;
        }

        /************************************** Softmax *************************************/
        Softmax::Softmax() {}

        // helper method to set number of threads
        void Softmax::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }

        Eigen::Tensor<double, 2> Softmax::forward(const Eigen::Tensor<double, 2>& input)
        {
            if (input.size() == 0)
            {
                throw std::runtime_error("SoftMax: Empty input tensor in 2D forward");
            }
            
            int batch_size = input.dimension(0);
            int num_classes = input.dimension(1);
            
            output_cache_2d_ = Eigen::Tensor<double, 2>(batch_size, num_classes);
            
            // Parallelize across batch dimension
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < batch_size; ++b)
            {
                // Find max for numerical stability (for this batch sample)
                double max_val = input(b, 0);
                for (int j = 1; j < num_classes; ++j)
                {
                    if (input(b, j) > max_val)
                        max_val = input(b, j);
                }
                
                // Compute exp(x - max) and sum
                double sum_exp = 0.0;
                for (int j = 0; j < num_classes; ++j)
                {
                    double exp_val = std::exp(input(b, j) - max_val);
                    output_cache_2d_(b, j) = exp_val;
                    sum_exp += exp_val;
                }
                
                // Normalize to get softmax
                sum_exp += 1e-15; // numerical stability
                for (int j = 0; j < num_classes; ++j)
                {
                    output_cache_2d_(b, j) /= sum_exp;
                }
            }
            
            return output_cache_2d_;
        }

        Eigen::Tensor<double, 2> Softmax::backward(const Eigen::Tensor<double, 2>& grad_output)
        {
            if (grad_output.dimension(0) != output_cache_2d_.dimension(0) ||
                grad_output.dimension(1) != output_cache_2d_.dimension(1) ||
                grad_output.size() == 0)
            {
                throw std::runtime_error("SoftMax: Shape mismatch or empty input in 2D backward");
            }
            
            int batch_size = grad_output.dimension(0);
            int num_classes = grad_output.dimension(1);
            
            Eigen::Tensor<double, 2> grad_input(batch_size, num_classes);
            
            // Parallelize across batch dimension
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < batch_size; ++b)
            {
                // Compute sum(grad_output * softmax) for this batch sample
                double sum_grad_softmax = 0.0;
                for (int j = 0; j < num_classes; ++j)
                {
                    sum_grad_softmax += grad_output(b, j) * output_cache_2d_(b, j);
                }
                
                // Compute gradient: softmax * (grad_output - sum)
                for (int j = 0; j < num_classes; ++j)
                {
                    grad_input(b, j) = output_cache_2d_(b, j) * (grad_output(b, j) - sum_grad_softmax);
                }
            }
            
            return grad_input;
        }

        Eigen::Tensor<double, 4> Softmax::forward(const Eigen::Tensor<double, 4>& input) 
        {
            // TODO: Implement Softmax: exp(x_i) / sum(exp(x_j)) for all j
            // Remember to subtract max for numerical stability
            Eigen::Tensor<double, 1> output(input.dimension(0));
            output.setZero();
            return output;
        }

        Eigen::Tensor<double, 4> Softmax::backward(const Eigen::Tensor<double, 4>& grad_output) 
        {
            // TODO: Implement Softmax derivative: s_i * (δ_ij - s_j) where s = softmax(x)
            Eigen::Tensor<double, 1> grad_input(grad_output.dimension(0));
            grad_input.setZero();
            return grad_input;
        }
    }
}