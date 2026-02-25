/**
 * @file conv2d.hpp
 * @brief 2D Convolutional layer
 */

#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "CppNet/layers/layer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Layers
    {
        /**
         * @class Conv2D
         * @brief 2D Convolutional layer with configurable kernel, stride, and padding.
         *
         * Input shape:  [batch, in_channels, height, width]
         * Output shape: [batch, out_channels, out_height, out_width]
         */
        class Conv2D : public Layer
        {
        public:
            /**
             * @param in_channels Number of input channels
             * @param out_channels Number of output filters
             * @param kernel_size Size of the convolving kernel (square)
             * @param stride Stride of the convolution (default 1)
             * @param padding Zero-padding added to both sides (default 0)
             * @param bias Whether to include a learnable bias (default true)
             * @param device Compute backend: "cpu-eigen", "cpu", or "gpu"
             */
            Conv2D(int in_channels, int out_channels, int kernel_size,
                   int stride = 1, int padding = 0, bool bias = true,
                   const std::string& device = "cpu-eigen");
            ~Conv2D();

            Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& input);
            Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& grad_output);

            bool is_trainable() const override { return trainable_; }
            void step(Optimizers::Optimizer& optimizer, float learning_rate) override;

            void freeze() { trainable_ = false; }
            void unfreeze() { trainable_ = true; }

            // Accessors
            Eigen::Tensor<float, 4>& get_weights() { return weights_; }
            const Eigen::Tensor<float, 4>& get_weights() const { return weights_; }
            Eigen::Tensor<float, 1>& get_biases() { return biases_; }
            Eigen::Tensor<float, 4>& get_grad_weights() { return grad_weights_; }
            Eigen::Tensor<float, 1>& get_grad_biases() { return grad_biases_; }

            int get_in_channels() const { return in_channels_; }
            int get_out_channels() const { return out_channels_; }
            int get_kernel_size() const { return kernel_size_; }
            int get_stride() const { return stride_; }
            int get_padding() const { return padding_; }

        private:
            int in_channels_;
            int out_channels_;
            int kernel_size_;
            int stride_;
            int padding_;
            bool bias_flag_;
            bool trainable_ = true;
            std::string device_;

            Eigen::Tensor<float, 4> weights_;      // [out_channels, in_channels, kH, kW]
            Eigen::Tensor<float, 1> biases_;        // [out_channels]
            Eigen::Tensor<float, 4> grad_weights_;
            Eigen::Tensor<float, 1> grad_biases_;
            Eigen::Tensor<float, 4> input_cache_;   // cached input for backward
        };
    }
}

#endif // CONV2D_HPP
