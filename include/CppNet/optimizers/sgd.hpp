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
#include "CppNet/layers/linear.hpp"
#include "CppNet/optimizers/optimizer.hpp"



namespace CppNet
{
    namespace Optimizers
    {
        class SGD : public Optimizer
        {
        public:
            SGD(int gpu_block_size = 256);
            void step(CppNet::Layers::Linear& layer, float learning_rate) override;
            void update(float* weights, const float* gradients,
                        int size, float learning_rate) override;

        private:
            int gpu_block_size_;
            void step_gpu(CppNet::Layers::Linear& layer, float learning_rate);
        };
    }
}

#endif // SGD_HPP