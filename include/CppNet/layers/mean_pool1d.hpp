/**
 * @file mean_pool1d.hpp
 * @brief Mean pooling over the sequence (time) dimension
 *
 * Reduces a 3D tensor [batch, seq_len, features] to [batch, features]
 * by averaging over the sequence dimension.
 *
 * Commonly used in Transformer / sequence models to aggregate
 * per-token representations into a fixed-length vector before
 * a classification head.
 */

#ifndef MEAN_POOL1D_HPP
#define MEAN_POOL1D_HPP

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "CppNet/kernels/gpu/gpu.hpp"
#endif

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class MeanPool1D
         * @brief Average pooling over the sequence dimension
         *
         * input:  [batch, seq_len, features]
         * output: [batch, features]
         */
        class MeanPool1D : public Layer
        {
        public:
            explicit MeanPool1D(const std::string& device = "cpu-eigen");
            ~MeanPool1D() override = default;

            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 3>& input);
            Eigen::Tensor<float, 3> backward(const Eigen::Tensor<float, 2>& grad_output);

            bool is_trainable() const override { return false; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;
            void reset_grads() override {}

        private:
            std::string device_;
            int batch_cache_ = 0;
            int seq_len_cache_ = 0;
            int features_cache_ = 0;
        };
    }
}

#endif // MEAN_POOL1D_HPP
