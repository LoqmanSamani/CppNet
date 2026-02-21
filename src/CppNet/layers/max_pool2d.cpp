/**
 * @file max_pool2d.cpp
 * @brief 2D Max Pooling layer implementation
 *
 * Forward: selects max value in each pooling window.
 * Backward: routes gradient only to the position of the max value.
 */

#include "CppNet/layers/max_pool2d.hpp"
#include <algorithm>
#include <limits>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Layers
    {
        MaxPool2D::MaxPool2D(int pool_size, int stride, const std::string& device)
            : pool_size_(pool_size),
              stride_(stride > 0 ? stride : pool_size),
              device_(device)
        {
        }

        const Eigen::Tensor<float, 4> MaxPool2D::forward(const Eigen::Tensor<float, 4>& input)
        {
            // input: [batch, channels, H, W]
            input_cache_ = input;

            int batch = input.dimension(0);
            int channels = input.dimension(1);
            int H = input.dimension(2);
            int W = input.dimension(3);

            int H_out = (H - pool_size_) / stride_ + 1;
            int W_out = (W - pool_size_) / stride_ + 1;

            Eigen::Tensor<float, 4> output(batch, channels, H_out, W_out);
            max_indices_.resize(batch, channels, H_out, W_out);

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2) if(device_ == "cpu")
            #endif
            for (int n = 0; n < batch; ++n)
            {
                for (int c = 0; c < channels; ++c)
                {
                    for (int oh = 0; oh < H_out; ++oh)
                    {
                        for (int ow = 0; ow < W_out; ++ow)
                        {
                            float max_val = -std::numeric_limits<float>::infinity();
                            int max_idx = 0;

                            for (int ph = 0; ph < pool_size_; ++ph)
                            {
                                for (int pw = 0; pw < pool_size_; ++pw)
                                {
                                    int ih = oh * stride_ + ph;
                                    int iw = ow * stride_ + pw;
                                    float val = input(n, c, ih, iw);

                                    if (val > max_val)
                                    {
                                        max_val = val;
                                        max_idx = ph * pool_size_ + pw;
                                    }
                                }
                            }

                            output(n, c, oh, ow) = max_val;
                            max_indices_(n, c, oh, ow) = max_idx;
                        }
                    }
                }
            }

            return output;
        }

        const Eigen::Tensor<float, 4> MaxPool2D::backward(const Eigen::Tensor<float, 4>& grad_output)
        {
            int batch = input_cache_.dimension(0);
            int channels = input_cache_.dimension(1);
            int H = input_cache_.dimension(2);
            int W = input_cache_.dimension(3);
            int H_out = grad_output.dimension(2);
            int W_out = grad_output.dimension(3);

            Eigen::Tensor<float, 4> grad_input(batch, channels, H, W);
            grad_input.setZero();

            for (int n = 0; n < batch; ++n)
                for (int c = 0; c < channels; ++c)
                    for (int oh = 0; oh < H_out; ++oh)
                        for (int ow = 0; ow < W_out; ++ow)
                        {
                            int idx = max_indices_(n, c, oh, ow);
                            int ph = idx / pool_size_;
                            int pw = idx % pool_size_;
                            int ih = oh * stride_ + ph;
                            int iw = ow * stride_ + pw;

                            grad_input(n, c, ih, iw) += grad_output(n, c, oh, ow);
                        }

            return grad_input;
        }
    }
}
