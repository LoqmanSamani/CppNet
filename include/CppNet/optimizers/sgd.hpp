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


// header file for SGD optimizer implementation

namespace CppNet
{
    namespace Optimizers
    {
        class SGD : public Optimizer
        {
        public:
            SGD(int gpu_block_size = 256);
            void step(CppNet::Layers::Linear& layer, float learning_rate) override;
            //void step_gpu(CppNet::Layers::Linear& layer, float learning_rate);// override;
            //void step(CppNet::Layers::Conv2d& layer, double learning_rate) override;
            //void step(CppNet::Layers::MultiHeadAttention& layer, double learning_rate) override;

        private:
            int gpu_block_size_;
            void step_gpu(CppNet::Layers::Linear& layer, float learning_rate);
        };
    }
}

#endif // SGD_HPP