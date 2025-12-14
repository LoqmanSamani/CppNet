

// The file "relu.hpp" contains the header class of ReLU activation function for both two and four dimension tensors

#ifndef RELU_HPP
#define RELU_HPP

#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "CppNet/activations/activation.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

namespace CppNet
{
namespace Activations
    {
        class ReLU : public Activation
        {
            public:
            
                ReLU(int size, std::string device = "gpu", bool relu_2d = true, int channels = 0, int gpu_block_size = 256);
                ~ReLU();
                Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& pre_activation) override;
                Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) override;
                Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& pre_activation) override;
                Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output) override;
                static void set_num_threads(int num_threads);
                std::string get_device() { return device_; }
                const Eigen::Tensor<float, 2>& get_output_cache_2d() const { return output_cache_2d_; }
                const Eigen::Tensor<float, 4>& get_output_cache_4d() const { return output_cache_4d_; }
                Eigen::Tensor<float, 2>& get_output_cache_2d() { return output_cache_2d_; }
                Eigen::Tensor<float, 4>& get_output_cache_4d() { return output_cache_4d_; }

                // set max batch size for GPU
                void set_max_batch_size(int max_batch_size);
                
                // GPU memory management
                #ifdef USE_CUDA

                    void init_gpu_buffers(int max_batch_size);
                    void cleanup_gpu_buffers();
                    void sync_output_cache_to_gpu();      // CPU -> GPU
                    //void sync_output_cache_4d_to_gpu();      // CPU -> GPU
                    void sync_output_cache_from_gpu();    // GPU -> CPU
                    //void sync_output_cache_4d_from_gpu();    // GPU -> CPU
                    
                    float* get_d_output_cache_2d_() { return d_output_cache_2d_; }
                    float* get_d_output_cache_4d_() { return d_output_cache_4d_; }
                    
                    bool is_gpu_initialized() const { return gpu_initialized_; }
        
                #endif
   
            private:
                int size_;
                std::string device_;
                bool relu_2d_;
                int channels_;
                int gpu_block_size_;
                Eigen::Tensor<float, 2> output_cache_2d_;
                Eigen::Tensor<float, 4> output_cache_4d_;

                #ifdef USE_CUDA
                    float* d_output_cache_2d_;
                    float* d_output_cache_4d_;
                    int gpu_max_batch_size_;
                    bool gpu_initialized_;
                #endif

                void forward_gpu(const Eigen::Tensor<float, 2>& pre_activation);
                void backward_gpu(const Eigen::Tensor<float, 2>& grad_output);
                void forward_gpu(const Eigen::Tensor<float, 4>& pre_activation);
                void backward_gpu(const Eigen::Tensor<float, 4>& grad_output);
        };
    }
}

#endif // RELU_HPP


// The code below is "relu.cpp" file with the complete implementation of the entire class.

#include <iostream>
#include <cmath>
#include <omp.h>
#include <Eigen/Dense>
#include "CppNet/activations/relu.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"




namespace CppNet
{
    namespace Activations
    {
        ReLU::ReLU(int size, std::string device, bool relu_2d, int channels, int gpu_block_size): 
        size_(size), device_(device), relu_2d_(relu_2d), channels_(channels), gpu_block_size_(gpu_block_size)
        {
            #ifdef USE_CUDA
                d_output_cache_2d_ = nullptr;
                d_output_cache_4d_ = nullptr;
                gpu_max_batch_size_ = 0;
                gpu_initialized_ = false;
            #endif
        }

        ReLU::~ReLU() 
        {
            #ifdef USE_CUDA

                cleanup_gpu_buffers();

            #endif
        }
        void ReLU::set_max_batch_size(int max_batch_size)
        {
            #ifdef USE_CUDA

                if (device_ == "gpu") 
                {
                    init_gpu_buffers(max_batch_size);
                }
                
            #endif
        }

        #ifdef USE_CUDA

            void ReLU::init_gpu_buffers(int max_batch_size)
            {
                if (gpu_initialized_) 
                {
                    cleanup_gpu_buffers();
                }
                
                gpu_max_batch_size_ = max_batch_size;
                
                std::cout << "Allocating GPU buffers (max batch: " << max_batch_size << ")" << std::endl;

                cudaError_t err;
                
                if (relu_2d_)
                {
                    size_t buffer_size = max_batch_size * size_ * sizeof(float);
                    err = cudaMalloc(&d_output_cache_2d_, buffer_size);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error("CUDA allocation failed: " + std::string(cudaGetErrorString(err)));
                    }
                }
                else
                {
                    size_t buffer_size = max_batch_size * channels_ * size_ * size_ * sizeof(float);
                    err = cudaMalloc(&d_output_cache_4d_, buffer_size);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error("CUDA allocation failed: " + std::string(cudaGetErrorString(err)));
                    }
                }
                
                gpu_initialized_ = true;
                std::cout << "GPU buffers allocated successfully." << std::endl;
            }

            void ReLU::cleanup_gpu_buffers()
            {
                if (!gpu_initialized_) 
                {
                    return;
                }
                
                if (d_output_cache_2d_) 
                {
                    cudaFree(d_output_cache_2d_);
                }
            
                if (d_output_cache_4d_)
                {
                    cudaFree(d_output_cache_4d_);
                } 
                
                d_output_cache_2d_ = nullptr;
                d_output_cache_4d_ = nullptr;
                
                gpu_initialized_ = false;
            }
            void ReLU::sync_output_cache_to_gpu()
            {
                if (!gpu_initialized_) 
                {
                    return;
                }
                if (relu_2d_)
                {
                    cudaMemcpy(d_output_cache_2d_, output_cache_2d_.data(), size_ * size_  * sizeof(float), cudaMemcpyHostToDevice);
                }
                else
                {
                    cudaMemcpy(d_output_cache_4d_, output_cache_4d_.data(), size_ * size_ * channels_ * channels_ * sizeof(float), cudaMemcpyHostToDevice);
                }
            }
            
            void ReLU::sync_output_cache_from_gpu()
            {
                if (!gpu_initialized_) 
                {
                    return;
                }
                
                cudaError_t err;
                
                if (relu_2d_)
                {
                    err = cudaMemcpy(output_cache_2d_.data(), d_output_cache_2d_, output_cache_2d_.size() * sizeof(float), cudaMemcpyDeviceToHost);
                }
                else
                {
                    err = cudaMemcpy(output_cache_4d_.data(), d_output_cache_4d_, output_cache_4d_.size() * sizeof(float), cudaMemcpyDeviceToHost);
                }
                
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("cudaMemcpy D2H failed: " + std::string(cudaGetErrorString(err)));
                }
            }

            void ReLU::forward_gpu(const Eigen::Tensor<float, 2>& pre_activation)
            {
                if (!gpu_initialized_)
                {
                    throw std::runtime_error("GPU buffers not initialized. Call set_max_batch_size() first.");
                }

                const int rows = pre_activation.dimension(0);
                const int cols = pre_activation.dimension(1);
                const int total = rows * cols;
                const size_t bytes = total * sizeof(float);

                // Check buffer capacity
                if (total > gpu_max_batch_size_ * size_)
                {
                    throw std::runtime_error("Input exceeds allocated GPU buffer size.");
                }

                float* d_pre_activation = nullptr;
                cudaError_t err = cudaMalloc(&d_pre_activation, bytes);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
                }

                err = cudaMemcpy(d_pre_activation, pre_activation.data(), bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    cudaFree(d_pre_activation);
                    throw std::runtime_error("cudaMemcpy H2D failed: " + std::string(cudaGetErrorString(err)));
                }

                const int block = gpu_block_size_;
                const int grid = (total + block - 1) / block;

                CppNet::Kernels::GPU::relu_kernel<<<grid, block>>>(d_pre_activation, d_output_cache_2d_, total);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    cudaFree(d_pre_activation);
                    throw std::runtime_error("ReLU kernel launch failed: " + std::string(cudaGetErrorString(err)));
                }

                cudaDeviceSynchronize();
                cudaFree(d_pre_activation);
            }

            void ReLU::forward_gpu(const Eigen::Tensor<float, 4>& pre_activation)
            {
                if (!gpu_initialized_)
                {
                    throw std::runtime_error("GPU buffers not initialized. Call set_max_batch_size() first.");
                }
                
                int batch = pre_activation.dimension(0);
                int channels = pre_activation.dimension(1);
                int height = pre_activation.dimension(2);
                int width = pre_activation.dimension(3);

                const int total = batch * channels * height * width;
                const size_t bytes = total * sizeof(float);
                
                if (total > gpu_max_batch_size_ * channels_ * size_ * size_)
                {
                    throw std::runtime_error("Input exceeds allocated GPU buffer size.");
                }

                float* d_pre_activation = nullptr;
                cudaError_t err = cudaMalloc(&d_pre_activation, bytes);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
                }

                err = cudaMemcpy(d_pre_activation, pre_activation.data(), bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    cudaFree(d_pre_activation);
                    throw std::runtime_error("cudaMemcpy H2D failed: " + std::string(cudaGetErrorString(err)));
                }

                const int block = gpu_block_size_; 
                const int grid = (total + block - 1) / block;

                CppNet::Kernels::GPU::relu_kernel<<<grid, block>>>(d_pre_activation, d_output_cache_4d_, total);

                err = cudaGetLastError();
                if (err != cudaSuccess) 
                {
                    cudaFree(d_pre_activation);
                    throw std::runtime_error("ReLU kernel launch failed: " + std::string(cudaGetErrorString(err)));
                }

                cudaDeviceSynchronize();
                cudaFree(d_pre_activation);
            }

            void ReLU::backward_gpu(const Eigen::Tensor<float, 2>& grad_output)
            {
                if (!gpu_initialized_)
                {
                    throw std::runtime_error("GPU buffers not initialized. Call set_max_batch_size() first.");
                }

                const int rows = grad_output.dimension(0);
                const int cols = grad_output.dimension(1);
                const int total = rows * cols;
                const size_t bytes = total * sizeof(float);

                float* d_grad_input = nullptr;
                float* d_grad_output = nullptr;
                
                cudaError_t err = cudaMalloc(&d_grad_input, bytes);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("cudaMalloc grad_input failed: " + std::string(cudaGetErrorString(err)));
                }
                
                err = cudaMalloc(&d_grad_output, bytes);
                if (err != cudaSuccess)
                {
                    cudaFree(d_grad_input);
                    throw std::runtime_error("cudaMalloc grad_output failed: " + std::string(cudaGetErrorString(err)));
                }

                if (total > gpu_max_batch_size_ * size_)
                {
                    cudaFree(d_grad_input);
                    cudaFree(d_grad_output);
                    throw std::runtime_error("Gradient exceeds allocated GPU buffer size.");
                }

                err = cudaMemcpy(d_grad_output, grad_output.data(), bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    cudaFree(d_grad_input);
                    cudaFree(d_grad_output);
                    throw std::runtime_error("cudaMemcpy grad_output failed: " + std::string(cudaGetErrorString(err)));
                }

                const int block = gpu_block_size_;
                const int grid = (total + block - 1) / block;

                CppNet::Kernels::GPU::relu_grad_kernel<<<grid, block>>>(d_grad_output, d_output_cache_2d_, d_grad_input, total);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    cudaFree(d_grad_input);
                    cudaFree(d_grad_output);
                    throw std::runtime_error("ReLU backward kernel failed: " + std::string(cudaGetErrorString(err)));
                }

                cudaDeviceSynchronize();
                cudaFree(d_grad_input);
                cudaFree(d_grad_output);
            }

            void ReLU::backward_gpu(const Eigen::Tensor<float, 4>& grad_output)
            {
                if (!gpu_initialized_)
                {
                    throw std::runtime_error("GPU buffers not initialized. Call set_max_batch_size() first.");
                }

                int batch = grad_output.dimension(0);
                int channels = grad_output.dimension(1);
                int height = grad_output.dimension(2);
                int width = grad_output.dimension(3);

                const int total = batch * channels * height * width;
                const size_t bytes = total * sizeof(float);

                float* d_grad_input = nullptr;
                cudaMalloc(&d_grad_input,  batch * channels * height * width * sizeof(float));

                float* d_grad_output = nullptr;
                cudaMalloc(&d_grad_output,  batch * channels * height * width * sizeof(float));

                if (total > gpu_max_batch_size_)
                {
                    cudaFree(d_grad_input);
                    cudaFree(d_grad_output);
                    throw std::runtime_error("Gradient exceeds allocated GPU buffer size.");
                }

                cudaError_t err = cudaMemcpy(d_grad_output, grad_output.data(), bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("cudaMemcpy dA failed: " + std::string(cudaGetErrorString(err)));
                }

                const int block = gpu_block_size_;
                const int grid = (total + block - 1) / block;

                CppNet::Kernels::GPU::relu_grad_kernel<<<grid, block>>>(d_grad_output, d_output_cache_4d_, d_grad_input, total);

                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("ReLU backward kernel failed: " + std::string(cudaGetErrorString(err)));
                }

                cudaDeviceSynchronize();
                cudaFree(d_grad_input);
                cudaFree(d_grad_output);
            }
        #endif

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

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(2) schedule(static)
                for (int i = 0; i < rows; ++i)
                {
                    for (int j = 0; j < cols; ++j) 
                    {
                        output_cache_2d_(i, j) = std::max(0.0f, pre_activation(i, j));
                    }
                }
            }
            else if (device_ == "cpu-eigen")
            {
                Eigen::Tensor<float, 2> mask = (pre_activation > 0.0).template cast<float>();
                output_cache_2d_ = pre_activation * mask;
            }
            else if (device_ == "gpu")
            {
                forward_gpu(pre_activation);
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
            
            if (device_ == "cpu")
            {
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
            }
            else if (device_ == "cpu-eigen")
            {
                output_cache_4d_ = pre_activation.cwiseMax(0.0f);
            }
            else if (device_ == "gpu")
            {
                forward_gpu(pre_activation);
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

            if (device_ == "cpu")
            {
                #pragma omp parallel for collapse(2) schedule(static)
                for (int i = 0; i < rows; ++i)
                {
                    for (int j = 0; j < cols; ++j) 
                    {
                        grad_input(i, j) = grad_output(i, j) * (output_cache_2d_(i, j) > 0.0f ? 1.0f : 0.0f);
                    }
                }
            }
            else if (device_ == "cpu-eigen")
            {
                Eigen::Tensor<float, 2> mask = (output_cache_2d_ > 0.0f).template cast<float>();
                grad_input = grad_output * mask;
            }
            else if (device_ == "gpu")
            {
                backward_gpu(grad_output);
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
            
            if (device_ == "cpu")
            {
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
            }
            else if (device_ == "cpu-eigen")
            {
                Eigen::Tensor<float, 4> mask = (output_cache_4d_ > 0.0f).template cast<float>();
                grad_input = grad_output * mask;
            }
            
            return grad_input;
        }
    }
}


// here is the gpu kernel implementation of the kernels needed inside relu (relu.cu and relu_grad.cu)
#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "CppNet/kernels/gpu/gpu.hpp"



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {

            __global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int total_elements)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < total_elements)
                {
                    float x = input[idx];
                    output[idx] = x > 0.0f ? x : 0.0f;
                }
            }
        }
    }
}

#pragma once
#include <cuda_runtime.h>



namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {
            __global__ void relu_backward_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int total_elements)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < total_elements)
                {
                    dZ[idx] = (Z[idx] > 0.0f) ? dA[idx] : 0.0f;
                }
            }
        } 
    } 
} 
