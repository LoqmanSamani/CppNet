/**
 * @file lstm.cpp
 * @brief LSTM layer implementation with full BPTT backward pass
 *
 * Forward:
 *   gates = x_t * W_ih + h_{t-1} * W_hh + bias      (4*hidden_size cols)
 *   i = sigmoid(gates[:, 0:H])
 *   f = sigmoid(gates[:, H:2H])
 *   g = tanh   (gates[:, 2H:3H])
 *   o = sigmoid(gates[:, 3H:4H])
 *   c_t = f ◦ c_{t-1} + i ◦ g
 *   h_t = o ◦ tanh(c_t)
 */

#include "CppNet/layers/lstm.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/utils/init.hpp"
#include <cmath>
#include <stdexcept>

#ifdef USE_CUDA
#include "CppNet/kernels/gpu/gpu.hpp"
#endif

namespace CppNet
{
    namespace Layers
    {
        static inline float sigmoid(float x)
        {
            return 1.0f / (1.0f + std::exp(-x));
        }

        LSTM::LSTM(int input_size, int hidden_size, bool return_sequences,
                   const std::string& device)
            : input_size_(input_size), hidden_size_(hidden_size),
              return_sequences_(return_sequences), device_(device)
        {
            int gate4 = 4 * hidden_size_;

            // xavier uniform with per-gate fan_out = hidden_size_
            float limit_ih = std::sqrt(6.0f / static_cast<float>(input_size_ + hidden_size_));
            float limit_hh = std::sqrt(6.0f / static_cast<float>(hidden_size_ + hidden_size_));

            W_ih_ = CppNet::Utils::uniform_init(input_size_, gate4, -limit_ih, limit_ih);
            W_hh_ = CppNet::Utils::uniform_init(hidden_size_, gate4, -limit_hh, limit_hh);
            bias_.resize(gate4);
            bias_.setZero();

            for (int d = hidden_size_; d < 2 * hidden_size_; ++d)
                bias_(d) = 1.0f;

            grad_W_ih_.resize(input_size_, gate4);
            grad_W_hh_.resize(hidden_size_, gate4);
            grad_bias_.resize(gate4);
            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        LSTM::~LSTM() = default;

        const Eigen::Tensor<float, 3> LSTM::forward(
            const Eigen::Tensor<float, 3>& input)
        {
            // input: [batch, seq_len, input_size]
            input_cache_ = input;
            int batch = input.dimension(0);
            int seq_len = input.dimension(1);

            h0_.resize(batch, hidden_size_);
            c0_.resize(batch, hidden_size_);
            h0_.setZero();
            c0_.setZero();

            caches_.clear();
            caches_.resize(seq_len);

            Eigen::Tensor<float, 3> output(batch, seq_len, hidden_size_);

            #ifdef USE_CUDA
            if (device_ == "gpu")
            {
                forward_gpu(input, output, batch, seq_len);
                return output;
            }
            #endif

            int gate4 = 4 * hidden_size_;

            Eigen::Tensor<float, 2> h_prev = h0_;
            Eigen::Tensor<float, 2> c_prev = c0_;

            Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

            for (int t = 0; t < seq_len; ++t)
            {
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input(n, t, d);

                // gates = x_t * W_ih + h_prev * W_hh + bias
                Eigen::Tensor<float, 2> gates = x_t.contract(W_ih_, contract_dims) +
                                                  h_prev.contract(W_hh_, contract_dims);

                Eigen::Tensor<float, 2> i_gate(batch, hidden_size_);
                Eigen::Tensor<float, 2> f_gate(batch, hidden_size_);
                Eigen::Tensor<float, 2> g_gate(batch, hidden_size_);
                Eigen::Tensor<float, 2> o_gate(batch, hidden_size_);
                Eigen::Tensor<float, 2> c_t(batch, hidden_size_);
                Eigen::Tensor<float, 2> h_t(batch, hidden_size_);
                Eigen::Tensor<float, 2> tanh_c(batch, hidden_size_);

                for (int n = 0; n < batch; ++n)
                {
                    for (int d = 0; d < hidden_size_; ++d)
                    {
                        i_gate(n, d) = sigmoid(gates(n, d + 0 * hidden_size_) +
                                               bias_(d + 0 * hidden_size_));
                        f_gate(n, d) = sigmoid(gates(n, d + 1 * hidden_size_) +
                                               bias_(d + 1 * hidden_size_));
                        g_gate(n, d) = std::tanh(gates(n, d + 2 * hidden_size_) +
                                                 bias_(d + 2 * hidden_size_));
                        o_gate(n, d) = sigmoid(gates(n, d + 3 * hidden_size_) +
                                               bias_(d + 3 * hidden_size_));

                        c_t(n, d) = f_gate(n, d) * c_prev(n, d) + i_gate(n, d) * g_gate(n, d);
                        tanh_c(n, d) = std::tanh(c_t(n, d));
                        h_t(n, d) = o_gate(n, d) * tanh_c(n, d);

                        output(n, t, d) = h_t(n, d);
                    }
                }

                caches_[t].i_gate = i_gate;
                caches_[t].f_gate = f_gate;
                caches_[t].g_gate = g_gate;
                caches_[t].o_gate = o_gate;
                caches_[t].c_t = c_t;
                caches_[t].h_t = h_t;
                caches_[t].tanh_c = tanh_c;

                h_prev = h_t;
                c_prev = c_t;
            }

            return output;
        }

        const Eigen::Tensor<float, 3> LSTM::backward(
            const Eigen::Tensor<float, 3>& grad_output)
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

            int gate4 = 4 * hidden_size_;
            Eigen::Tensor<float, 2> dh_next(batch, hidden_size_);
            Eigen::Tensor<float, 2> dc_next(batch, hidden_size_);
            dh_next.setZero();
            dc_next.setZero();

            for (int t = seq_len - 1; t >= 0; --t)
            {
                auto& cache = caches_[t];

                // dh = grad_output[:, t, :] + dh_next
                Eigen::Tensor<float, 2> dh(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dh(n, d) = grad_output(n, t, d) + dh_next(n, d);

                // dc = dh * o * (1 - tanh_c^2) + dc_next
                Eigen::Tensor<float, 2> dc(batch, hidden_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dc(n, d) = dh(n, d) * cache.o_gate(n, d) *
                                   (1.0f - cache.tanh_c(n, d) * cache.tanh_c(n, d)) +
                                   dc_next(n, d);

                Eigen::Tensor<float, 2> dgates(batch, gate4);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                    {
                        // di = dc * g * i * (1 - i)
                        float di = dc(n, d) * cache.g_gate(n, d) *
                                   cache.i_gate(n, d) * (1.0f - cache.i_gate(n, d));
                        // df = dc * c_{t-1} * f * (1 - f)
                        float c_prev_val = (t > 0) ? caches_[t - 1].c_t(n, d) : 0.0f;
                        float df = dc(n, d) * c_prev_val *
                                   cache.f_gate(n, d) * (1.0f - cache.f_gate(n, d));
                        // dg = dc * i * (1 - g^2)
                        float dg = dc(n, d) * cache.i_gate(n, d) *
                                   (1.0f - cache.g_gate(n, d) * cache.g_gate(n, d));
                        // do = dh * tanh_c * o * (1 - o)
                        float do_ = dh(n, d) * cache.tanh_c(n, d) *
                                    cache.o_gate(n, d) * (1.0f - cache.o_gate(n, d));

                        dgates(n, d + 0 * hidden_size_) = di;
                        dgates(n, d + 1 * hidden_size_) = df;
                        dgates(n, d + 2 * hidden_size_) = dg;
                        dgates(n, d + 3 * hidden_size_) = do_;
                    }

                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input_cache_(n, t, d);

                Eigen::Tensor<float, 2> h_prev = (t > 0) ? caches_[t - 1].h_t : h0_;

                Eigen::array<Eigen::IndexPair<int>, 1> contract_batch = {Eigen::IndexPair<int>(0, 0)};
                grad_W_ih_ += x_t.contract(dgates, contract_batch);      // [input, 4H]
                grad_W_hh_ += h_prev.contract(dgates, contract_batch);   // [hidden, 4H]

                for (int g = 0; g < gate4; ++g)
                    for (int n = 0; n < batch; ++n)
                        grad_bias_(g) += dgates(n, g);

                // grad_input[:, t, :] = dgates * W_ih^T
                Eigen::array<Eigen::IndexPair<int>, 1> contract_gate = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> dx = dgates.contract(W_ih_, contract_gate);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        grad_input(n, t, d) = dx(n, d);

                // dh_next = dgates * W_hh^T
                dh_next = dgates.contract(W_hh_, contract_gate);

                // dc_next = dc * f_gate
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < hidden_size_; ++d)
                        dc_next(n, d) = dc(n, d) * cache.f_gate(n, d);
            }

            return grad_input;
        }

        void LSTM::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            if (!trainable_) return;

            optimizer.update(W_ih_.data(), grad_W_ih_.data(), W_ih_.size(), learning_rate);
            optimizer.update(W_hh_.data(), grad_W_hh_.data(), W_hh_.size(), learning_rate);
            optimizer.update(bias_.data(), grad_bias_.data(), bias_.size(), learning_rate);

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        void LSTM::reset_grads()
        {
            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        #ifdef USE_CUDA
        void LSTM::forward_gpu(const Eigen::Tensor<float, 3>& input,
                               Eigen::Tensor<float, 3>& output,
                               int batch, int seq_len)
        {
            int I = input_size_, H = hidden_size_;
            int gate4 = 4 * H;

            float *d_W_ih, *d_W_hh, *d_bias;
            float *d_x_t, *d_h_prev, *d_c_prev;
            float *d_pre_ih, *d_pre_hh;
            float *d_i, *d_f, *d_g, *d_o, *d_c_t, *d_h_t, *d_tanh_c;

            cudaMalloc(&d_W_ih,    I * gate4 * sizeof(float));
            cudaMalloc(&d_W_hh,    H * gate4 * sizeof(float));
            cudaMalloc(&d_bias,    gate4 * sizeof(float));
            cudaMalloc(&d_x_t,     batch * I * sizeof(float));
            cudaMalloc(&d_h_prev,  batch * H * sizeof(float));
            cudaMalloc(&d_c_prev,  batch * H * sizeof(float));
            cudaMalloc(&d_pre_ih,  batch * gate4 * sizeof(float));
            cudaMalloc(&d_pre_hh,  batch * gate4 * sizeof(float));
            cudaMalloc(&d_i,       batch * H * sizeof(float));
            cudaMalloc(&d_f,       batch * H * sizeof(float));
            cudaMalloc(&d_g,       batch * H * sizeof(float));
            cudaMalloc(&d_o,       batch * H * sizeof(float));
            cudaMalloc(&d_c_t,     batch * H * sizeof(float));
            cudaMalloc(&d_h_t,     batch * H * sizeof(float));
            cudaMalloc(&d_tanh_c,  batch * H * sizeof(float));

            cudaMemcpy(d_W_ih, W_ih_.data(), I * gate4 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_W_hh, W_hh_.data(), H * gate4 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bias, bias_.data(), gate4 * sizeof(float),       cudaMemcpyHostToDevice);
            cudaMemset(d_h_prev, 0, batch * H * sizeof(float));
            cudaMemset(d_c_prev, 0, batch * H * sizeof(float));

            Eigen::Tensor<float, 2> x_t(batch, I);

            for (int t = 0; t < seq_len; ++t)
            {
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < I; ++d)
                        x_t(n, d) = input(n, t, d);
                cudaMemcpy(d_x_t, x_t.data(), batch * I * sizeof(float), cudaMemcpyHostToDevice);

                // pre_ih = x_t * W_ih  [batch, 4H]
                dim3 block(32, 32);
                dim3 grid((gate4 + 31) / 32, (batch + 31) / 32);
                Kernels::GPU::matmul_kernel<<<grid, block>>>(d_x_t, d_W_ih, d_pre_ih, batch, gate4, I);

                // pre_hh = h_prev * W_hh  [batch, 4H]
                Kernels::GPU::matmul_kernel<<<grid, block>>>(d_h_prev, d_W_hh, d_pre_hh, batch, gate4, H);

                int total = batch * H;
                int bk = 256;
                int gr = (total + bk - 1) / bk;
                Kernels::GPU::lstm_gates_forward_kernel<<<gr, bk>>>(
                    d_pre_ih, d_pre_hh, d_bias, d_c_prev,
                    d_i, d_f, d_g, d_o, d_c_t, d_h_t, d_tanh_c,
                    batch, H);

                Eigen::Tensor<float, 2> buf(batch, H);
                cudaMemcpy(buf.data(), d_i, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].i_gate = buf;
                cudaMemcpy(buf.data(), d_f, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].f_gate = buf;
                cudaMemcpy(buf.data(), d_g, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].g_gate = buf;
                cudaMemcpy(buf.data(), d_o, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].o_gate = buf;
                cudaMemcpy(buf.data(), d_c_t, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].c_t = buf;
                cudaMemcpy(buf.data(), d_h_t, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].h_t = buf;
                cudaMemcpy(buf.data(), d_tanh_c, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].tanh_c = buf;

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        output(n, t, d) = caches_[t].h_t(n, d);

                cudaMemcpy(d_h_prev, d_h_t, batch * H * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_c_prev, d_c_t, batch * H * sizeof(float), cudaMemcpyDeviceToDevice);
            }

            cudaFree(d_W_ih); cudaFree(d_W_hh); cudaFree(d_bias);
            cudaFree(d_x_t); cudaFree(d_h_prev); cudaFree(d_c_prev);
            cudaFree(d_pre_ih); cudaFree(d_pre_hh);
            cudaFree(d_i); cudaFree(d_f); cudaFree(d_g); cudaFree(d_o);
            cudaFree(d_c_t); cudaFree(d_h_t); cudaFree(d_tanh_c);
        }

        void LSTM::backward_gpu(const Eigen::Tensor<float, 3>& grad_output,
                                Eigen::Tensor<float, 3>& grad_input,
                                int batch, int seq_len)
        {
            int I = input_size_, H = hidden_size_;
            int gate4 = 4 * H;

            float *d_W_ih, *d_W_hh;
            float *d_x_t, *d_h_prev, *d_h_t;
            float *d_i, *d_f, *d_g, *d_o, *d_tanh_c, *d_c_prev;
            float *d_dh, *d_dh_next, *d_dc_next, *d_dc_out;
            float *d_dgates, *d_dgates_t;
            float *d_dW_ih, *d_dW_hh, *d_dbias;
            float *d_dW_ih_t, *d_dW_hh_t;
            float *d_dx;

            cudaMalloc(&d_W_ih,     I * gate4 * sizeof(float));
            cudaMalloc(&d_W_hh,     H * gate4 * sizeof(float));
            cudaMalloc(&d_x_t,      batch * I * sizeof(float));
            cudaMalloc(&d_h_prev,   batch * H * sizeof(float));
            cudaMalloc(&d_h_t,      batch * H * sizeof(float));
            cudaMalloc(&d_i,        batch * H * sizeof(float));
            cudaMalloc(&d_f,        batch * H * sizeof(float));
            cudaMalloc(&d_g,        batch * H * sizeof(float));
            cudaMalloc(&d_o,        batch * H * sizeof(float));
            cudaMalloc(&d_tanh_c,   batch * H * sizeof(float));
            cudaMalloc(&d_c_prev,   batch * H * sizeof(float));
            cudaMalloc(&d_dh,       batch * H * sizeof(float));
            cudaMalloc(&d_dh_next,  batch * H * sizeof(float));
            cudaMalloc(&d_dc_next,  batch * H * sizeof(float));
            cudaMalloc(&d_dc_out,   batch * H * sizeof(float));
            cudaMalloc(&d_dgates,   batch * gate4 * sizeof(float));
            cudaMalloc(&d_dgates_t, batch * gate4 * sizeof(float));
            cudaMalloc(&d_dW_ih,    I * gate4 * sizeof(float));
            cudaMalloc(&d_dW_hh,    H * gate4 * sizeof(float));
            cudaMalloc(&d_dW_ih_t,  I * gate4 * sizeof(float));
            cudaMalloc(&d_dW_hh_t,  H * gate4 * sizeof(float));
            cudaMalloc(&d_dbias,    gate4 * sizeof(float));
            cudaMalloc(&d_dx,       batch * I * sizeof(float));

            cudaMemcpy(d_W_ih, W_ih_.data(), I * gate4 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_W_hh, W_hh_.data(), H * gate4 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_dW_ih,   0, I * gate4 * sizeof(float));
            cudaMemset(d_dW_hh,   0, H * gate4 * sizeof(float));
            cudaMemset(d_dbias,   0, gate4 * sizeof(float));
            cudaMemset(d_dh_next, 0, batch * H * sizeof(float));
            cudaMemset(d_dc_next, 0, batch * H * sizeof(float));

            Eigen::Tensor<float, 2> x_t(batch, I);
            Eigen::Tensor<float, 2> grad_t(batch, H);

            for (int t = seq_len - 1; t >= 0; --t)
            {
                auto& cache = caches_[t];

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        grad_t(n, d) = grad_output(n, t, d);
                cudaMemcpy(d_dh, grad_t.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);

                int total = batch * H;
                int bk = 256;
                Kernels::GPU::elementwise_kernel<<<(total + bk - 1) / bk, bk>>>(d_dh, d_dh_next, d_dh, total, 0, 0);

                cudaMemcpy(d_i,      cache.i_gate.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_f,      cache.f_gate.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_g,      cache.g_gate.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_o,      cache.o_gate.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_tanh_c, cache.tanh_c.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);

                if (t > 0)
                    cudaMemcpy(d_c_prev, caches_[t - 1].c_t.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                else
                    cudaMemset(d_c_prev, 0, batch * H * sizeof(float));

                Kernels::GPU::lstm_gates_backward_kernel<<<(total + bk - 1) / bk, bk>>>(
                    d_dh, d_dc_next, d_i, d_f, d_g, d_o, d_tanh_c, d_c_prev,
                    d_dgates_t, d_dc_out, batch, H);

                // dc_next = dc_out for next iteration
                cudaMemcpy(d_dc_next, d_dc_out, batch * H * sizeof(float), cudaMemcpyDeviceToDevice);

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < I; ++d)
                        x_t(n, d) = input_cache_(n, t, d);
                cudaMemcpy(d_x_t, x_t.data(), batch * I * sizeof(float), cudaMemcpyHostToDevice);

                if (t > 0)
                    cudaMemcpy(d_h_prev, caches_[t - 1].h_t.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                else
                    cudaMemset(d_h_prev, 0, batch * H * sizeof(float));

                dim3 block(32, 32);
                int ws = I * gate4;
                cudaMemset(d_dW_ih_t, 0, ws * sizeof(float));
                dim3 grid_ih((gate4 + 31) / 32, (I + 31) / 32);
                Kernels::GPU::matmul_grad_weights_kernel<<<grid_ih, block>>>(d_x_t, d_dgates_t, d_dW_ih_t, batch, I, gate4);
                Kernels::GPU::elementwise_kernel<<<(ws + bk - 1) / bk, bk>>>(d_dW_ih, d_dW_ih_t, d_dW_ih, ws, 0, 0);

                int wsh = H * gate4;
                cudaMemset(d_dW_hh_t, 0, wsh * sizeof(float));
                dim3 grid_hh((gate4 + 31) / 32, (H + 31) / 32);
                Kernels::GPU::matmul_grad_weights_kernel<<<grid_hh, block>>>(d_h_prev, d_dgates_t, d_dW_hh_t, batch, H, gate4);
                Kernels::GPU::elementwise_kernel<<<(wsh + bk - 1) / bk, bk>>>(d_dW_hh, d_dW_hh_t, d_dW_hh, wsh, 0, 0);

                Kernels::GPU::bias_grad_kernel<<<gate4, 256>>>(d_dgates_t, d_dbias, batch, gate4);

                // grad_input = dgates * W_ih^T
                dim3 grid_dx((I + 31) / 32, (batch + 31) / 32);
                cudaMemset(d_dx, 0, batch * I * sizeof(float));
                Kernels::GPU::matmul_grad_input_kernel<<<grid_dx, block>>>(d_dgates_t, d_W_ih, d_dx, batch, I, gate4);
                Eigen::Tensor<float, 2> dx(batch, I);
                cudaMemcpy(dx.data(), d_dx, batch * I * sizeof(float), cudaMemcpyDeviceToHost);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < I; ++d)
                        grad_input(n, t, d) = dx(n, d);

                // dh_next = dgates * W_hh^T
                dim3 grid_dh((H + 31) / 32, (batch + 31) / 32);
                cudaMemset(d_dh_next, 0, batch * H * sizeof(float));
                Kernels::GPU::matmul_grad_input_kernel<<<grid_dh, block>>>(d_dgates_t, d_W_hh, d_dh_next, batch, H, gate4);
            }

            cudaMemcpy(grad_W_ih_.data(), d_dW_ih, I * gate4 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_W_hh_.data(), d_dW_hh, H * gate4 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_bias_.data(), d_dbias, gate4 * sizeof(float),       cudaMemcpyDeviceToHost);

            cudaFree(d_W_ih); cudaFree(d_W_hh);
            cudaFree(d_x_t); cudaFree(d_h_prev); cudaFree(d_h_t);
            cudaFree(d_i); cudaFree(d_f); cudaFree(d_g); cudaFree(d_o);
            cudaFree(d_tanh_c); cudaFree(d_c_prev);
            cudaFree(d_dh); cudaFree(d_dh_next); cudaFree(d_dc_next); cudaFree(d_dc_out);
            cudaFree(d_dgates); cudaFree(d_dgates_t);
            cudaFree(d_dW_ih); cudaFree(d_dW_hh); cudaFree(d_dW_ih_t); cudaFree(d_dW_hh_t);
            cudaFree(d_dbias); cudaFree(d_dx);
        }
        #endif // USE_CUDA
    }
}
