#include <cmath>
#include <omp.h>
#include <chrono> 
#include <Eigen/Dense>
#include "CppNet/utils/init.hpp"
#include "CppNet/optimizers/sgd.hpp"
#include "CppNet/activations/activation.hpp" 
#include "CppNet/activations/relu.hpp"
#include "CppNet/activations/sigmoid.hpp" 
#include "CppNet/kernels/gpu/gpu.hpp"
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
            // initialize GPU-specific members
            #ifdef USE_CUDA
                
                d_weights_ = nullptr;
                d_bias_ = nullptr;
                d_input_cache_ = nullptr;
                d_output_ = nullptr;
                d_grad_weights_ = nullptr;
                d_grad_bias_ = nullptr;
                d_grad_input_ = nullptr;
                gpu_initialized_ = false;
                gpu_max_batch_size_ = 0;

            #endif

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

        Linear::~Linear() 
        {
            #ifdef USE_CUDA

                cleanup_gpu_buffers();

            #endif
        }

        void Linear::set_max_batch_size(int max_batch_size)
        {
            #ifdef USE_CUDA

                if (device_ == "gpu") 
                {
                    init_gpu_buffers(max_batch_size);
                }
                
            #endif
        }


        #ifdef USE_CUDA
        
            // GPU buffer initialization
            void Linear::init_gpu_buffers(int max_batch_size)
            {
                // clean up old buffers if they exist
                if (gpu_initialized_) 
                {
                    cleanup_gpu_buffers();
                }
                
                gpu_max_batch_size_ = max_batch_size;
                
                std::cout << "Allocating GPU buffers for layer '" << layer_name_ 
                        << "' (max batch: " << max_batch_size << ")" << std::endl;
                
                // allocate persistent GPU memory
                cudaMalloc(&d_weights_, in_size_ * out_size_ * sizeof(float));
                if (bias_) 
                {
                    cudaMalloc(&d_bias_, out_size_ * sizeof(float));
                    cudaMalloc(&d_grad_bias_, out_size_ * sizeof(float));
                }
                
                // allocate buffers for largest expected batch
                cudaMalloc(&d_input_cache_, max_batch_size * in_size_ * sizeof(float));
                cudaMalloc(&d_output_, max_batch_size * out_size_ * sizeof(float));
                cudaMalloc(&d_grad_weights_, in_size_ * out_size_ * sizeof(float));
                cudaMalloc(&d_grad_input_, max_batch_size * in_size_ * sizeof(float));
                
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) 
                {
                    throw std::runtime_error("CUDA allocation failed for layer '" + layer_name_ + "': " + cudaGetErrorString(err));
                }
                
                // transfer weights to GPU
                sync_weights_to_gpu();
                
                gpu_initialized_ = true;
                
                std::cout << "GPU buffers allocated successfully for layer '" << layer_name_ << "'" << std::endl;
            }

            // GPU buffer clean up
            void Linear::cleanup_gpu_buffers()
            {
                if (!gpu_initialized_) 
                {
                    return;
                }
                
                if (d_weights_) 
                {
                    cudaFree(d_weights_);
                }
            
                if (d_bias_)
                {
                    cudaFree(d_bias_);
                } 
                if (d_input_cache_)
                {
                    cudaFree(d_input_cache_);
                } 
                if (d_output_)
                {
                    cudaFree(d_output_);
                } 
                if (d_grad_weights_)
                {
                    cudaFree(d_grad_weights_);
                } 

                if (d_grad_bias_) 
                {
                    cudaFree(d_grad_bias_);
                }
                if (d_grad_input_) 
                {
                    cudaFree(d_grad_input_);
                }
                
                d_weights_ = nullptr;
                d_bias_ = nullptr;
                d_input_cache_ = nullptr;
                d_output_ = nullptr;
                d_grad_weights_ = nullptr;
                d_grad_bias_ = nullptr;
                d_grad_input_ = nullptr;
                
                gpu_initialized_ = false;
            }

            // sync weigths CPU -> GPU
            void Linear::sync_weights_to_gpu()
            {
                if (!gpu_initialized_) 
                {
                    return;
                }
                
                cudaMemcpy(d_weights_, weights_.data(), in_size_ * out_size_ * sizeof(float), cudaMemcpyHostToDevice);
                
                if (bias_) 
                {
                    cudaMemcpy(d_bias_, biases_.data(), out_size_ * sizeof(float), cudaMemcpyHostToDevice);
                }
            }

            // sync gradients GPU -> CPU 
            void Linear::sync_gradients_from_gpu()
            {
                if (!gpu_initialized_ || !trainable_) 
                {
                    return;
                }
                
                cudaMemcpy(grad_weights_.data(), d_grad_weights_, in_size_ * out_size_ * sizeof(float), cudaMemcpyDeviceToHost);
                
                if (bias_) 
                {
                    cudaMemcpy(grad_biases_.data(), d_grad_bias_, out_size_ * sizeof(float), cudaMemcpyDeviceToHost);
                }
            }

            void Linear::forward_gpu(
                    const Eigen::Tensor<float, 2>& input, const Eigen::Tensor<float, 2>& weights_,
                    const Eigen::Tensor<float, 1>& biases_, Eigen::Tensor<float, 2>& output,
                    int batch_size, int input_size, int output_size, bool bias_) 
            {
                if (!gpu_initialized_) 
                {
                    std::cerr << "Error: GPU buffers not initialized for layer '" 
                            << layer_name_ << "'. Call set_max_batch_size() first.\n";
                    return;
                }

                if (batch_size > gpu_max_batch_size_) 
                {
                    throw std::runtime_error(
                        "Batch size " + std::to_string(batch_size) +
                        " exceeds GPU buffer size " + std::to_string(gpu_max_batch_size_) +
                        " in layer '" + layer_name_ + "'"
                    );
                }

                cudaMemcpy(d_input_cache_, input.data(), batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

                dim3 block(32, 32);
                dim3 grid((output_size + 31) / 32, (batch_size  + 31) / 32);

                CppNet::Kernels::GPU::matmul_kernel<<<grid, block>>>(d_input_cache_, d_weights_, d_output_, batch_size, output_size, input_size);

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) 
                {
                    throw std::runtime_error("CUDA matmul kernel error: " + std::string(cudaGetErrorString(err)));
                }

                if (bias_) 
                {
                    dim3 bias_block(16, 16);
                    dim3 bias_grid((output_size + 15) / 16, (batch_size  + 15) / 16);

                    CppNet::Kernels::GPU::add_bias_kernel<<<bias_grid, bias_block>>>(d_output_, d_bias_, batch_size, output_size);

                    err = cudaGetLastError();
                    if (err != cudaSuccess) 
                    {
                        throw std::runtime_error("CUDA bias kernel error: " + std::string(cudaGetErrorString(err)));
                    }
                }

                cudaMemcpy(output.data(), d_output_, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
            }


            void Linear::backward_gpu(
                const Eigen::Tensor<float, 2>& grad_output, const Eigen::Tensor<float, 2>& in_cache_,
                const Eigen::Tensor<float, 2>& weights_, Eigen::Tensor<float, 2>& grad_weights_, 
                Eigen::Tensor<float, 1>& grad_biases_, Eigen::Tensor<float, 2>& grad_input,  
                int batch_size, int output_size, int input_size, bool trainable_, bool bias_)
            {
                if (!gpu_initialized_) 
                {
                    std::cerr << "Error: GPU buffers not initialized." << std::endl;
                    return;
                }
                
                float* d_grad_output;
                cudaMalloc(&d_grad_output, batch_size * output_size * sizeof(float));
                cudaMemcpy(d_grad_output, grad_output.data(), batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
                
                if (trainable_) 
                {
                    cudaMemset(d_grad_weights_, 0, input_size * output_size * sizeof(float));
                    if (bias_) 
                    {
                        cudaMemset(d_grad_bias_, 0, output_size * sizeof(float));
                    }
                    
                    // weight gradient: dW = X^T * dY
                    dim3 block(32, 32);
                    dim3 grid((output_size + 31) / 32, (input_size + 31) / 32);
                    
                    CppNet::Kernels::GPU::matmul_grad_weights_kernel<<<grid, block>>>(d_input_cache_, d_grad_output, d_grad_weights_, batch_size, input_size, output_size);
                    
                    // bias gradient: sum over batch
                    if (bias_) 
                    {
                        dim3 bias_block(256);
                        dim3 bias_grid(output_size);
                        CppNet::Kernels::GPU::bias_grad_kernel<<<bias_grid, bias_block>>>(d_grad_output, d_grad_bias_, batch_size, output_size);
                    }
                }
                
                // input gradient: dX = dY * W^T 
                cudaMemset(d_grad_input_, 0, batch_size * input_size * sizeof(float));
                
                dim3 block(32, 32);
                dim3 grid((input_size + 31) / 32, (batch_size + 31) / 32);
                
                CppNet::Kernels::GPU::matmul_grad_input_kernel<<<grid, block>>>(d_grad_output, d_weights_, d_grad_input_, batch_size, input_size, output_size);
                
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) 
                {
                    throw std::runtime_error("CUDA kernel error in backward pass of layer '" + layer_name_ + "': " + cudaGetErrorString(err));
                }
                
                cudaMemcpy(grad_input.data(), d_grad_input_, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost);
                
                if (trainable_) 
                {
                    sync_gradients_from_gpu();
                }
                
                cudaFree(d_grad_output);
            }

        #endif

        void Linear::init_params_and_grads()
        {
            // initialize parameters and gradients
            CppNet::Utils::Initialization init(in_size_, out_size_);
            init.init_params(weights_, weight_init_);
            init.init_params(grad_weights_, "zeros");
            if (bias_)
            {
                init.init_params(biases_, "zeros");
                init.init_params(grad_biases_, "zeros");
            }
            else
            {
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
                weight_init_ = old_method;
                throw std::runtime_error("Failed to reinitialize with method '" + new_init_method + "': " + e.what());
            }
        }

        // optimizer step
        void Linear::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            optimizer.step(*this, learning_rate);
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
                forward_cpu(input, weights_, biases_, output, batch_size, input_size, output_size, bias_);
            }
            else if (should_parallelize && device_ == "gpu")
            {
                forward_gpu(input, weights_, biases_, output, batch_size, input_size, output_size, bias_);
            }
            else if (!should_parallelize)
            {
                forward_eigen(input, weights_, biases_, output, batch_size, input_size, output_size, bias_);
            }
            
            return output;
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
                backward_cpu(grad_output, in_cache_, weights_, grad_weights_, grad_biases_, grad_input,
                     batch_size, output_size, input_size, trainable_, bias_);
            }
            else if (should_parallelize && device_ == "gpu")
            {
                backward_gpu(grad_output, in_cache_, weights_, grad_weights_, grad_biases_, grad_input,
                     batch_size, output_size, input_size, trainable_, bias_);

            }
            else if (!should_parallelize)
            {
                backward_eigen(grad_output, in_cache_, weights_, grad_weights_, grad_biases_, grad_input,
                     batch_size, output_size, input_size, trainable_, bias_);
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