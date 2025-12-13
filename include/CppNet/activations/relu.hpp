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
                const Eigen::Tensor<float, 2>& get_output_cache_2d() { output_cache_2d_; }
                const Eigen::Tensor<float, 4>& get_output_cache_4d() { output_cache_4d_; }
                Eigen::Tensor<float, 2>& get_output_cache_2d() const { output_cache_2d_; }
                Eigen::Tensor<float, 4>& get_output_cache_4d() const { output_cache_4d_; }

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
                    float* get_d_output_cache_2d_() { return d_output_cache_4d_; }
                    
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