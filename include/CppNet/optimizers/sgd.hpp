#ifndef SGD_HPP
#define SGD_HPP


#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>
#include <memory>
#include "CppNet/optimizers/optimizer.hpp"



namespace CppNet
{
    namespace Optimizers
    {
        class SGD : public Optimizer
        {
        public:
            SGD(const std::string& device = "cpu", int gpu_block_size = 256);
            ~SGD();
            void update(float* weights, const float* gradients,
                        int size, float learning_rate) override;

        private:
            std::string device_;
            int gpu_block_size_;

#ifdef USE_CUDA
            float* d_weights_ = nullptr;
            float* d_gradients_ = nullptr;
            int gpu_buf_ = 0;

            void ensure_gpu(int size);
            void release_gpu();
            void update_gpu(float* weights, const float* gradients,
                            int size, float learning_rate);
#endif
        };
    }
}

#endif // SGD_HPP