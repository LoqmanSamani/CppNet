#include <cmath>
#include <omp.h>
#include <chrono> 
#include <Eigen/Dense>
#include "CppNet/optimizers/sgd.hpp"  
#include "CppNet/activations/relu.hpp"
#include "CppNet/activations/sigmoid.hpp" 
#include "CppNet/kernels/gpu.hpp"
#include "CppNet/layers/linear.hpp"




namespace CppNet
{
    namespace Layers
    {
        //********************* Linear (Fully Connected: Dense) Layer *********************//

        Linear::Linear(
            int in_size, 
            int out_size, 
            std::string layer_name, 
            bool trainable, 
            bool bias, 
            std::string device,
            std::string weight_init,
            int parallel_threshold
        ) 
        : in_size_(in_size), out_size_(out_size), layer_name_(layer_name), 
          trainable_(trainable), bias_(bias), device_(device), weight_init_(weight_init),
          parallel_threshold_(parallel_threshold)
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

        // initialize parameters
        void Linear::init_params_and_grads()
        {
            std::random_device rd;
            double scale = 0.0;
            double mean = 0.0;
            double std_dev = 0.0;
            bool use_normal = false;  // flag to determine distribution type
            
            // calculate initialization parameters based on method
            if (weight_init_ == "xavier" || weight_init_ == "xavier_uniform")
            {
                // xavier/glorot uniform: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
                scale = std::sqrt(6.0 / (in_size_ + out_size_));
                use_normal = false;
            }
            else if (weight_init_ == "xavier_normal" || weight_init_ == "glorot_normal")
            {
                // xavier/glorot normal: N(0, sqrt(2/(fan_in + fan_out)))
                mean = 0.0;
                std_dev = std::sqrt(2.0 / (in_size_ + out_size_));
                use_normal = true;
            }
            else if (weight_init_ == "he" || weight_init_ == "he_uniform")
            {
                // he uniform (for relu): U(-sqrt(6/fan_in), sqrt(6/fan_in))
                scale = std::sqrt(6.0 / in_size_);
                use_normal = false;
            }
            else if (weight_init_ == "he_normal")
            {
                // he normal (for relu): N(0, sqrt(2/fan_in))
                mean = 0.0;
                std_dev = std::sqrt(2.0 / in_size_);
                use_normal = true;
            }
            else if (weight_init_ == "lecun_uniform")
            {
                // lecun uniform: U(-sqrt(3/fan_in), sqrt(3/fan_in))
                scale = std::sqrt(3.0 / in_size_);
                use_normal = false;
            }
            else if (weight_init_ == "lecun_normal")
            {
                // lecun normal: N(0, sqrt(1/fan_in))
                mean = 0.0;
                std_dev = std::sqrt(1.0 / in_size_);
                use_normal = true;
            }
            else if (weight_init_ == "uniform")
            {
                // simple uniform distribution: U(-0.1, 0.1)
                scale = 0.1;
                use_normal = false;
            }
            else if (weight_init_ == "normal")
            {
                // simple normal distribution: N(0, 0.01)
                mean = 0.0;
                std_dev = 0.01;
                use_normal = true;
            }
            else if (weight_init_ == "zeros")
            {
                // initialize with zeros (useful for some specific architectures)
                scale = 0.0;
                use_normal = false;
            }
            else if (weight_init_ == "ones")
            {
                // initialize with ones (rarely used, but available)
                scale = -1.0; // special flag for ones initialization
                use_normal = false;
            }
            else
            {
                throw std::runtime_error("Unknown weight initialization method: '" + weight_init_ + 
                                        "' in layer: " + layer_name_ + 
                                        "\nSupported methods: xavier, xavier_normal, he, he_normal, " +
                                        "lecun_uniform, lecun_normal, uniform, normal, zeros, ones");
            }

            // initialize weight tensor
            weights_ = Eigen::Tensor<float, 2>(in_size_, out_size_);

            // only parallelize for larger matrices to avoid overhead
            const int total_elements = in_size_ * out_size_;
            const bool should_parallelize = (total_elements > parallel_threshold_);

            if (weight_init_ == "zeros")
            {
                // special case: zero initialization
                weights_.setZero();
            }
            else if (weight_init_ == "ones")
            {
                // special case: ones initialization
                weights_.setConstant(1.0);
            }
            else if (should_parallelize)
            {
                // parallel initialization for large matrices
                #pragma omp parallel
                {
                    // each thread gets its own random generator to avoid race conditions
                    std::mt19937 local_gen(rd() + omp_get_thread_num() * 1000 + std::chrono::high_resolution_clock::now().time_since_epoch().count() % 1000);
                    
                    if (use_normal)
                    {
                        std::normal_distribution<float> local_dist(mean, std_dev);
                        #pragma omp for collapse(2)
                        for (int i = 0; i < in_size_; ++i)
                        {
                            for (int j = 0; j < out_size_; ++j)
                            {
                                weights_(i, j) = local_dist(local_gen);
                            }
                        }
                    }
                    else
                    {
                        std::uniform_real_distribution<float> local_dist(-scale, scale);
                        #pragma omp for collapse(2)
                        for (int i = 0; i < in_size_; ++i)
                        {
                            for (int j = 0; j < out_size_; ++j)
                            {
                                weights_(i, j) = local_dist(local_gen);
                            }
                        }
                    }
                }
            }
            else
            {
                // serial initialization for small matrices
                std::mt19937 gen(rd());
                
                if (use_normal)
                {
                    std::normal_distribution<float> dist(mean, std_dev);
                    for (int i = 0; i < in_size_; ++i)
                    {
                        for (int j = 0; j < out_size_; ++j)
                        {
                            weights_(i, j) = dist(gen);
                        }
                    }
                }
                else
                {
                    std::uniform_real_distribution<float> dist(-scale, scale);
                    for (int i = 0; i < in_size_; ++i)
                    {
                        for (int j = 0; j < out_size_; ++j)
                        {
                            weights_(i, j) = dist(gen);
                        }
                    }
                }
            }

            // initialize weight gradients with zeros
            grad_weights_ = Eigen::Tensor<float, 2>(in_size_, out_size_);
            grad_weights_.setZero();

            // initialize biases and bias gradients
            if (bias_)
            {
                biases_ = Eigen::Tensor<float, 1>(out_size_);
                grad_biases_ = Eigen::Tensor<float, 1>(out_size_);
                
                // biases are typically initialized to zero regardless of weight initialization
                biases_.setZero();
                grad_biases_.setZero();
            }
            else
            {
                // initialize empty tensors when bias is disabled
                biases_ = Eigen::Tensor<float, 1>(0);
                grad_biases_ = Eigen::Tensor<float, 1>(0);
            }
        }

        // reinitialize parameters
        void Linear::reinitialize_weights(const std::string& new_init_method)
        {
            std::string old_method = weight_init_;
            weight_init_ = new_init_method;
            
            try
            {
                init_params_and_grads();
                std::cout << "Layer '" << layer_name_ << "' weights reinitialized from '" 
                        << old_method << "' to '" << new_init_method << "'" << std::endl;
            }
            catch (const std::exception& e)
            {
                // restore old method if new one fails
                weight_init_ = old_method;
                throw std::runtime_error("Failed to reinitialize with method '" + new_init_method + "': " + e.what());
            }
        }

        // optimizer step
        void Linear::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            optimizer.step(*this, learning_rate);
        }

        void Linear::forward_gpu(
            const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_, 
            const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output, 
            int batch_size, int input_size, int output_size, bool bias_)
        {
            
            CppNet::Kernels::GPU::matmul_gpu(input.data(), weights_.data(), output.data(), batch_size, output_size, input_size);
        
            if (bias_)
            {
                CppNet::Kernels::GPU::add_bias_gpu(output.data(), biases_.data(), batch_size, output_size);
            }
        }

        void Linear::forward_cpu(
            const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_, 
            const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output, 
            int batch_size, int input_size, int output_size, bool bias_)
        {
            // manual matrix multiplication with openmp
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < batch_size; ++b)
            {
                for (int j = 0; j < output_size; ++j) 
                {
                    float sum = 0.0;
                    // vectorize inner loop and use reduction for better performance
                    #pragma omp simd reduction(+:sum)
                    for (int i = 0; i < input_size; ++i) 
                    {
                        sum += input(b, i) * weights_(i, j);
                    }
                    output(b, j) = sum;
                }
            }

            // add bias if enabled
            if (bias_) 
            {
                #pragma omp parallel for collapse(2)
                for (int b = 0; b < batch_size; ++b) 
                {
                    for (int j = 0; j < output_size; ++j) 
                    {
                        output(b, j) += biases_(j);
                    }
                }
            }
        }

        void Linear::forward_eigen(
            const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_, 
            const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output, 
            int batch_size, int input_size, int output_size, bool bias_)
        {
            // forward calculation
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
            output = input.contract(weights_, product_dims);

            if (bias_) 
            {
                // add bias to each row - broadcasting the bias vector
                Eigen::array<Eigen::Index, 2> broadcast_dims({input.dimension(0), 1});
                Eigen::Tensor<float, 2> bias_broadcasted = biases_.reshape(Eigen::array<Eigen::Index, 2>({1, biases_.dimension(0)})).broadcast(broadcast_dims);
                output = output + bias_broadcasted;
            }
        }


        // forward pass 
        Eigen::Tensor<float, 2> Linear::forward(const Eigen::Tensor<float, 2>& input) 
        {
            // check dimensions
            if (input.dimension(1) != weights_.dimension(0))
            {
                throw std::runtime_error("Shape mismatch: in layer: " + layer_name_ + " input.dimension(1) must be equal weights_.dimension(0)!");
            }
            // store input to use later in gradient calculation
            in_cache_ = input;

            // get dimensions
            const int batch_size = input.dimension(0);
            const int input_size = input.dimension(1);
            const int output_size = weights_.dimension(1);

            // create output tensor
            Eigen::Tensor<float, 2> output(batch_size, output_size);

            
            const int total_elements = batch_size * output_size;
            const bool should_parallelize = (total_elements > parallel_threshold_);

            if (should_parallelize && device_ == "cpu")
            {
                Linear::forward_cpu(input, weights_, biases_, output, batch_size, input_size, output_size, bias_);
            }
            else if (should_parallelize && device_ == "gpu")
            {
                Linear::forward_gpu(input, weights_, biases_, output, batch_size, input_size, output_size, bias_);
            }
            else if (!should_parallelize)
            {
                Linear::forward_eigen(input, weights_, biases_, output, batch_size, input_size, output_size, bias_);
            }
            
            return output;
        }

        void Linear::backward_gpu(
            const Eigen::Tensor<float, 2>& grad_output, const Eigen::Tensor<float, 2>& in_cache_,
            const Eigen::Tensor<float, 2>& weights_, Eigen::Tensor<float, 2>& grad_weights_, 
            Eigen::Tensor<float, 1>& grad_biases_, Eigen::Tensor<float, 2>& grad_input,  
            int batch_size, int output_size, int input_size, bool trainable_, bool bias_)
        {
            if (trainable_)
            {
                CppNet::Kernels::GPU::matmul_grad_weights_gpu(in_cache_.data(), grad_output.data(), grad_weights_.data(), batch_size, input_size, output_size);
                if (bias_)
                {
                    CppNet::Kernels::GPU::bias_grad_gpu(grad_output.data(), grad_biases_.data(), batch_size, output_size);
                }
            }

            // use gpu kernel
            CppNet::Kernels::GPU::matmul_grad_input_gpu(grad_output.data(), weights_.data(), grad_input.data(), batch_size, input_size, output_size);
            
        }

        void Linear::backward_cpu(
            const Eigen::Tensor<float, 2>& grad_output, const Eigen::Tensor<float, 2>& in_cache_,
            const Eigen::Tensor<float, 2>& weights_, Eigen::Tensor<float, 2>& grad_weights_, 
            Eigen::Tensor<float, 1>& grad_biases_, Eigen::Tensor<float, 2>& grad_input,  
            int batch_size, int output_size, int input_size, bool trainable_, bool bias_)
        {
            if (trainable_)
            {
                // manual computation with openmp: grad_weights_(i,j) = sum_b(in_cache_(b,i) * grad_output(b,j))
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < input_size; ++i) 
                {
                    for (int j = 0; j < output_size; ++j) 
                    {
                        float sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (int b = 0; b < batch_size; ++b) 
                        {
                            sum += in_cache_(b, i) * grad_output(b, j);
                        }
                        grad_weights_(i, j) += sum;
                    }
                }
                
                // gradient w.r.t. biases: sum over batch dimension
                if (bias_) 
                {
                    #pragma omp parallel for
                    for (int j = 0; j < output_size; ++j) 
                    {
                        float sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (int b = 0; b < batch_size; ++b) 
                        {
                            sum += grad_output(b, j);
                        }
                        grad_biases_(j) += sum;
                    }
                }
            }
            
            // manual computation with openmp: grad_input(b,i) = sum_j(grad_output(b,j) * weights_(i,j))
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < batch_size; ++b) 
            {
                for (int i = 0; i < input_size; ++i) 
                {
                    float sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (int j = 0; j < output_size; ++j) 
                    {
                        sum += grad_output(b, j) * weights_(i, j);
                    }
                    grad_input(b, i) += sum;
                }
            }
            
        }

        void Linear::backward_eigen(
            const Eigen::Tensor<float, 2>& grad_output, const Eigen::Tensor<float, 2>& in_cache_,
            const Eigen::Tensor<float, 2>& weights_, Eigen::Tensor<float, 2>& grad_weights_, 
            Eigen::Tensor<float, 1>& grad_biases_, Eigen::Tensor<float, 2>& grad_input,  
            int batch_size, int output_size, int input_size, bool trainable_, bool bias_)
        {
            if (trainable_)
            {
                // gradient w.r.t. weights: X^T * grad_out
                Eigen::array<int, 2> transpose_dims({1, 0});
                Eigen::Tensor<float, 2> X_transposed = in_cache_.shuffle(transpose_dims);
                
                Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
                grad_weights_ += X_transposed.contract(grad_output, product_dims);
                
                // gradient w.r.t. biases: sum over batch dimension
                if (bias_) 
                {
                    Eigen::array<int, 1> batch_dim({0});
                    grad_biases_ += grad_output.sum(batch_dim);
                }
            }

            Eigen::array<int, 2> transpose_dims({1, 0});
            Eigen::Tensor<float, 2> weights_transposed = weights_.shuffle(transpose_dims);
            
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
            grad_input += grad_output.contract(weights_transposed, product_dims);
        }

        Eigen::Tensor<float, 2> Linear::backward(const Eigen::Tensor<float, 2>& grad_output) 
        {
            // get dimensions
            const int batch_size = grad_output.dimension(0);
            const int output_size = grad_output.dimension(1);
            const int input_size = in_cache_.dimension(1);


            // dimension validation
            if (grad_output.dimension(0) != in_cache_.dimension(0)) 
            {
                throw std::runtime_error("Batch size mismatch in layer: " + layer_name_);
            }
            if (grad_output.dimension(1) != out_size_) 
            {
                throw std::runtime_error("Output size mismatch in layer: " + layer_name_);
            }

            Eigen::Tensor<float, 2> grad_input(batch_size, input_size);

            const int total_elements = batch_size * output_size;
            const bool should_parallelize = (total_elements > parallel_threshold_);
            
            if (should_parallelize && device_ == "cpu")
            {
                Linear::backward_cpu(grad_output, in_cache_, grad_weights_, grad_biases_, grad_input, 
                    batch_size, output_size, input_size, trainable_, bias_)
            }
            else if (should_parallelize && device_ == "gpu")
            {
                Linear::backward_gpu(grad_output, in_cache_, grad_weights_, grad_biases_, grad_input, 
                    batch_size, output_size, input_size, trainable_, bias_)

            }
            else if (!should_parallelize)
            {
                Linear::backward_eigen(grad_output, in_cache_, grad_weights_, grad_biases_, grad_input, 
                    batch_size, output_size, input_size, trainable_, bias_)
            }
                   
            return grad_input;
        }


        // helper method to set number of threads
        void Linear::set_num_threads(int num_threads) 
        {
            if (num_threads > 0) 
            {
                omp_set_num_threads(num_threads);
            }
        }
    }
}