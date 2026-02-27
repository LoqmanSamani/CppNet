/**
 * @file rnn.cpp
 * @brief Vanilla RNN layer implementation
 *
 * h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
 * Backward: BPTT (backpropagation through time)
 */

#include "CppNet/layers/rnn.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/utils/init.hpp"
#include <cmath>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "CppNet/kernels/gpu/gpu.hpp"
#endif

namespace CppNet
{
    namespace Layers
    {
        RNN::RNN(int input_size, int hidden_size, bool return_sequences,
                 const std::string& device)
            : input_size_(input_size), hidden_size_(hidden_size),
              return_sequences_(return_sequences), device_(device)
        {
            // Xavier initialization
            W_ih_ = CppNet::Utils::xavier_uniform(input_size_, hidden_size_);
            W_hh_ = CppNet::Utils::xavier_uniform(hidden_size_, hidden_size_);
            bias_.resize(hidden_size_);
            bias_.setZero();

            grad_W_ih_.resize(input_size_, hidden_size_);
            grad_W_hh_.resize(hidden_size_, hidden_size_);
            grad_bias_.resize(hidden_size_);
            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        RNN::~RNN() = default;

        const Eigen::Tensor<float, 3> RNN::forward(const Eigen::Tensor<float, 3>& input)
        {
            // input: [batch, seq_len, input_size]
            input_cache_ = input;
            int batch = input.dimension(0);
            int seq_len = input.dimension(1);

            hidden_states_.clear();
            hidden_states_.reserve(seq_len + 1);

            // h_0 = zeros
            Eigen::Tensor<float, 2> h_prev(batch, hidden_size_);
            h_prev.setZero();
            hidden_states_.push_back(h_prev);

            Eigen::Tensor<float, 3> output(batch, seq_len, hidden_size_);

            #ifdef USE_CUDA
            if (device_ == "gpu")
            {
                forward_gpu(input, output, batch, seq_len);
                return output;
            }
            #endif

            Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

            for (int t = 0; t < seq_len; ++t)
            {
                // extract x_t: [batch, input_size]
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input(n, t, d);

                // h_t = tanh(x_t * W_ih + h_{t-1} * W_hh + b)
                Eigen::Tensor<float, 2> pre_act = x_t.contract(W_ih_, contract_dims) +
                                                   h_prev.contract(W_hh_, contract_dims);

                Eigen::Tensor<float, 2> h_t(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                    {
                        h_t(n, d) = std::tanh(pre_act(n, d) + bias_(d));
                        output(n, t, d) = h_t(n, d);
                    }

                hidden_states_.push_back(h_t);
                h_prev = h_t;
            }

            return output;
        }

        const Eigen::Tensor<float, 3> RNN::backward(const Eigen::Tensor<float, 3>& grad_output)
        {
            // grad_output: [batch, seq_len, hidden_size]
            int batch = grad_output.dimension(0);
            int seq_len = grad_output.dimension(1);

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();

            Eigen::Tensor<float, 3> grad_input(batch, seq_len, input_size_);
            grad_input.setZero();

            #ifdef USE_CUDA
            if (device_ == "gpu")
            {
                backward_gpu(grad_output, grad_input, batch, seq_len);
                return grad_input;
            }
            #endif

            Eigen::Tensor<float, 2> dh_next(batch, hidden_size_);
            dh_next.setZero();

            for (int t = seq_len - 1; t >= 0; --t)
            {
                // dh = grad_output[:, t, :] + dh_next
                Eigen::Tensor<float, 2> dh(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dh(n, d) = grad_output(n, t, d) + dh_next(n, d);

                // dtanh = dh * (1 - h_t^2)
                auto& h_t = hidden_states_[t + 1];
                Eigen::Tensor<float, 2> dtanh(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dtanh(n, d) = dh(n, d) * (1.0f - h_t(n, d) * h_t(n, d));

                // extract x_t and h_{t-1}
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input_cache_(n, t, d);

                auto& h_prev = hidden_states_[t];

                // grad_W_ih += x_t^T * dtanh
                Eigen::array<Eigen::IndexPair<int>, 1> contract_batch = {Eigen::IndexPair<int>(0, 0)};
                grad_W_ih_ += x_t.contract(dtanh, contract_batch);
                grad_W_hh_ += h_prev.contract(dtanh, contract_batch);

                // grad_bias += sum(dtanh, axis=0)
                for (int d = 0; d < hidden_size_; ++d)
                    for (int n = 0; n < batch; ++n)
                        grad_bias_(d) += dtanh(n, d);

                // grad_input[:, t, :] = dtanh * W_ih^T
                Eigen::array<Eigen::IndexPair<int>, 1> contract_hidden = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> dx = dtanh.contract(W_ih_, contract_hidden);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        grad_input(n, t, d) = dx(n, d);

                // dh_next = dtanh * W_hh^T
                dh_next = dtanh.contract(W_hh_, contract_hidden);
            }

            return grad_input;
        }

        void RNN::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            if (!trainable_) return;

            optimizer.update(W_ih_.data(), grad_W_ih_.data(), W_ih_.size(), learning_rate);
            optimizer.update(W_hh_.data(), grad_W_hh_.data(), W_hh_.size(), learning_rate);
            optimizer.update(bias_.data(), grad_bias_.data(), bias_.size(), learning_rate);

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        void RNN::reset_grads()
        {
            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        #ifdef USE_CUDA
        void RNN::forward_gpu(const Eigen::Tensor<float, 3>& input,
                              Eigen::Tensor<float, 3>& output,
                              int batch, int seq_len)
        {
            int I = input_size_, H = hidden_size_;

            float *d_W_ih, *d_W_hh, *d_bias;
            float *d_x_t, *d_h_prev, *d_pre_ih, *d_pre_hh, *d_h_t;

            cudaMalloc(&d_W_ih,   I * H * sizeof(float));
            cudaMalloc(&d_W_hh,   H * H * sizeof(float));
            cudaMalloc(&d_bias,   H * sizeof(float));
            cudaMalloc(&d_x_t,    batch * I * sizeof(float));
            cudaMalloc(&d_h_prev, batch * H * sizeof(float));
            cudaMalloc(&d_pre_ih, batch * H * sizeof(float));
            cudaMalloc(&d_pre_hh, batch * H * sizeof(float));
            cudaMalloc(&d_h_t,    batch * H * sizeof(float));

            cudaMemcpy(d_W_ih, W_ih_.data(), I * H * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_W_hh, W_hh_.data(), H * H * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bias, bias_.data(), H * sizeof(float),      cudaMemcpyHostToDevice);
            cudaMemset(d_h_prev, 0, batch * H * sizeof(float));

            Eigen::Tensor<float, 2> x_t(batch, I);

            for (int t = 0; t < seq_len; ++t)
            {
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < I; ++d)
                        x_t(n, d) = input(n, t, d);
                cudaMemcpy(d_x_t, x_t.data(), batch * I * sizeof(float), cudaMemcpyHostToDevice);

                // pre_ih = x_t * W_ih
                dim3 block(32, 32);
                dim3 grid((H + 31) / 32, (batch + 31) / 32);
                Kernels::GPU::matmul_kernel<<<grid, block>>>(d_x_t, d_W_ih, d_pre_ih, batch, H, I);

                // pre_hh = h_prev * W_hh
                Kernels::GPU::matmul_kernel<<<grid, block>>>(d_h_prev, d_W_hh, d_pre_hh, batch, H, H);

                // h_t = tanh(pre_ih + pre_hh + bias)
                int total = batch * H;
                int bk = 256;
                int gr = (total + bk - 1) / bk;
                Kernels::GPU::rnn_tanh_forward_kernel<<<gr, bk>>>(d_pre_ih, d_pre_hh, d_bias, d_h_t, total, H);

                Eigen::Tensor<float, 2> h_t_cpu(batch, H);
                cudaMemcpy(h_t_cpu.data(), d_h_t, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                hidden_states_.push_back(h_t_cpu);

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        output(n, t, d) = h_t_cpu(n, d);

                // h_prev = h_t (device-to-device)
                cudaMemcpy(d_h_prev, d_h_t, batch * H * sizeof(float), cudaMemcpyDeviceToDevice);
            }

            cudaFree(d_W_ih); cudaFree(d_W_hh); cudaFree(d_bias);
            cudaFree(d_x_t);  cudaFree(d_h_prev);
            cudaFree(d_pre_ih); cudaFree(d_pre_hh); cudaFree(d_h_t);
        }

        void RNN::backward_gpu(const Eigen::Tensor<float, 3>& grad_output,
                               Eigen::Tensor<float, 3>& grad_input,
                               int batch, int seq_len)
        {
            int I = input_size_, H = hidden_size_;

            float *d_W_ih, *d_W_hh;
            float *d_x_t, *d_h_prev, *d_h_t, *d_dtanh;
            float *d_grad_t, *d_dh_next;
            float *d_dW_ih_t, *d_dW_hh_t;
            float *d_dW_ih, *d_dW_hh, *d_dbias;
            float *d_dx;

            cudaMalloc(&d_W_ih,     I * H * sizeof(float));
            cudaMalloc(&d_W_hh,     H * H * sizeof(float));
            cudaMalloc(&d_x_t,      batch * I * sizeof(float));
            cudaMalloc(&d_h_prev,   batch * H * sizeof(float));
            cudaMalloc(&d_h_t,      batch * H * sizeof(float));
            cudaMalloc(&d_dtanh,    batch * H * sizeof(float));
            cudaMalloc(&d_grad_t,   batch * H * sizeof(float));
            cudaMalloc(&d_dh_next,  batch * H * sizeof(float));
            cudaMalloc(&d_dW_ih_t,  I * H * sizeof(float));
            cudaMalloc(&d_dW_hh_t,  H * H * sizeof(float));
            cudaMalloc(&d_dW_ih,    I * H * sizeof(float));
            cudaMalloc(&d_dW_hh,    H * H * sizeof(float));
            cudaMalloc(&d_dbias,    H * sizeof(float));
            cudaMalloc(&d_dx,       batch * I * sizeof(float));

            cudaMemcpy(d_W_ih, W_ih_.data(), I * H * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_W_hh, W_hh_.data(), H * H * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_dW_ih,   0, I * H * sizeof(float));
            cudaMemset(d_dW_hh,   0, H * H * sizeof(float));
            cudaMemset(d_dbias,   0, H * sizeof(float));
            cudaMemset(d_dh_next, 0, batch * H * sizeof(float));

            Eigen::Tensor<float, 2> x_t(batch, I);
            Eigen::Tensor<float, 2> grad_t(batch, H);

            for (int t = seq_len - 1; t >= 0; --t)
            {
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        grad_t(n, d) = grad_output(n, t, d);
                cudaMemcpy(d_grad_t, grad_t.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);

                cudaMemcpy(d_h_t, hidden_states_[t + 1].data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);

                // dtanh = (grad_t + dh_next) * (1 - h_t^2)
                int total = batch * H;
                int bk = 256;
                int gr = (total + bk - 1) / bk;
                Kernels::GPU::rnn_tanh_backward_kernel<<<gr, bk>>>(d_grad_t, d_dh_next, d_h_t, d_dtanh, total, H);

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < I; ++d)
                        x_t(n, d) = input_cache_(n, t, d);
                cudaMemcpy(d_x_t, x_t.data(), batch * I * sizeof(float), cudaMemcpyHostToDevice);

                cudaMemcpy(d_h_prev, hidden_states_[t].data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);

                // dW_ih_t = x_t^T * dtanh, then accumulate
                dim3 block(32, 32);
                int wsize = I * H;
                cudaMemset(d_dW_ih_t, 0, wsize * sizeof(float));
                dim3 grid_w((H + 31) / 32, (I + 31) / 32);
                Kernels::GPU::matmul_grad_weights_kernel<<<grid_w, block>>>(d_x_t, d_dtanh, d_dW_ih_t, batch, I, H);
                Kernels::GPU::elementwise_kernel<<<(wsize + 255) / 256, 256>>>(d_dW_ih, d_dW_ih_t, d_dW_ih, wsize, 0, 0);

                // dW_hh_t = h_prev^T * dtanh, then accumulate
                int whsize = H * H;
                cudaMemset(d_dW_hh_t, 0, whsize * sizeof(float));
                dim3 grid_wh((H + 31) / 32, (H + 31) / 32);
                Kernels::GPU::matmul_grad_weights_kernel<<<grid_wh, block>>>(d_h_prev, d_dtanh, d_dW_hh_t, batch, H, H);
                Kernels::GPU::elementwise_kernel<<<(whsize + 255) / 256, 256>>>(d_dW_hh, d_dW_hh_t, d_dW_hh, whsize, 0, 0);

                // dbias += sum(dtanh, axis=0)
                Kernels::GPU::bias_grad_kernel<<<H, 256>>>(d_dtanh, d_dbias, batch, H);

                // dx = dtanh * W_ih^T
                dim3 grid_dx((I + 31) / 32, (batch + 31) / 32);
                cudaMemset(d_dx, 0, batch * I * sizeof(float));
                Kernels::GPU::matmul_grad_input_kernel<<<grid_dx, block>>>(d_dtanh, d_W_ih, d_dx, batch, I, H);
                Eigen::Tensor<float, 2> dx(batch, I);
                cudaMemcpy(dx.data(), d_dx, batch * I * sizeof(float), cudaMemcpyDeviceToHost);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < I; ++d)
                        grad_input(n, t, d) = dx(n, d);

                // dh_next = dtanh * W_hh^T
                dim3 grid_dh((H + 31) / 32, (batch + 31) / 32);
                cudaMemset(d_dh_next, 0, batch * H * sizeof(float));
                Kernels::GPU::matmul_grad_input_kernel<<<grid_dh, block>>>(d_dtanh, d_W_hh, d_dh_next, batch, H, H);
            }

            cudaMemcpy(grad_W_ih_.data(), d_dW_ih, I * H * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_W_hh_.data(), d_dW_hh, H * H * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_bias_.data(), d_dbias, H * sizeof(float),      cudaMemcpyDeviceToHost);

            cudaFree(d_W_ih); cudaFree(d_W_hh);
            cudaFree(d_x_t); cudaFree(d_h_prev); cudaFree(d_h_t); cudaFree(d_dtanh);
            cudaFree(d_grad_t); cudaFree(d_dh_next);
            cudaFree(d_dW_ih_t); cudaFree(d_dW_hh_t);
            cudaFree(d_dW_ih); cudaFree(d_dW_hh); cudaFree(d_dbias);
            cudaFree(d_dx);
        }
        #endif // USE_CUDA
    }
}
