/**
 * @file gru.cpp
 * @brief GRU layer implementation with BPTT backward pass
 *
 * Forward (per timestep):
 *   z_t = sigmoid(x_t * W_iz + h_{t-1} * W_hz + b_z)        update gate
 *   r_t = sigmoid(x_t * W_ir + h_{t-1} * W_hr + b_r)        reset gate
 *   n_t = tanh   (x_t * W_in + (r_t ◦ h_{t-1}) * W_hn + b_n)  candidate
 *   h_t = (1 - z_t) ◦ n_t + z_t ◦ h_{t-1}
 */

#include "CppNet/layers/gru.hpp"
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/utils/init.hpp"
#include <cmath>

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

        GRU::GRU(int input_size, int hidden_size, bool return_sequences,
                 const std::string& device)
            : input_size_(input_size), hidden_size_(hidden_size),
              return_sequences_(return_sequences), device_(device)
        {
            int gate3 = 3 * hidden_size_;

            // Xavier uniform with per-gate fan_out = hidden_size_
            float limit_ih = std::sqrt(6.0f / static_cast<float>(input_size_ + hidden_size_));
            float limit_hh = std::sqrt(6.0f / static_cast<float>(hidden_size_ + hidden_size_));

            W_ih_ = CppNet::Utils::uniform_init(input_size_, gate3, -limit_ih, limit_ih);
            W_hh_ = CppNet::Utils::uniform_init(hidden_size_, gate3, -limit_hh, limit_hh);
            bias_.resize(gate3);
            bias_.setZero();

            grad_W_ih_.resize(input_size_, gate3);
            grad_W_hh_.resize(hidden_size_, gate3);
            grad_bias_.resize(gate3);
            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        GRU::~GRU() = default;

        const Eigen::Tensor<float, 3> GRU::forward(
            const Eigen::Tensor<float, 3>& input)
        {
            input_cache_ = input;
            int batch = input.dimension(0);
            int seq_len = input.dimension(1);
            int H = hidden_size_;

            h0_.resize(batch, H);
            h0_.setZero();

            caches_.clear();
            caches_.resize(seq_len);

            Eigen::Tensor<float, 3> output(batch, seq_len, H);

            #ifdef USE_CUDA
            if (device_ == "gpu") { forward_gpu(input, output, batch, seq_len); return output; }
            #endif

            Eigen::Tensor<float, 2> h_prev = h0_;

            Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

            for (int t = 0; t < seq_len; ++t)
            {
                // extract x_t [batch, input_size]
                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input(n, t, d);

                // compute all gates from concatenated weights
                Eigen::Tensor<float, 2> x_gates = x_t.contract(W_ih_, contract_dims);
                Eigen::Tensor<float, 2> h_gates = h_prev.contract(W_hh_, contract_dims);

                Eigen::Tensor<float, 2> z_gate(batch, H);
                Eigen::Tensor<float, 2> r_gate(batch, H);
                Eigen::Tensor<float, 2> n_cand(batch, H);
                Eigen::Tensor<float, 2> rh(batch, H);
                Eigen::Tensor<float, 2> h_t(batch, H);

                for (int n = 0; n < batch; ++n)
                {
                    for (int d = 0; d < H; ++d)
                    {
                        // z = sigmoid(x_gates_z + h_gates_z + b_z)
                        z_gate(n, d) = sigmoid(
                            x_gates(n, d + 0 * H) + h_gates(n, d + 0 * H) +
                            bias_(d + 0 * H));

                        // r = sigmoid(x_gates_r + h_gates_r + b_r)
                        r_gate(n, d) = sigmoid(
                            x_gates(n, d + 1 * H) + h_gates(n, d + 1 * H) +
                            bias_(d + 1 * H));

                        // rh = r ◦ h_prev
                        rh(n, d) = r_gate(n, d) * h_prev(n, d);
                    }
                }

                Eigen::Tensor<float, 2> W_hn(hidden_size_, H);
                for (int i = 0; i < hidden_size_; ++i)
                    for (int j = 0; j < H; ++j)
                        W_hn(i, j) = W_hh_(i, j + 2 * H);

                Eigen::Tensor<float, 2> rh_proj = rh.contract(W_hn, contract_dims);

                for (int n = 0; n < batch; ++n)
                {
                    for (int d = 0; d < H; ++d)
                    {
                        // n = tanh(x_gates_n + rh_proj + b_n)
                        n_cand(n, d) = std::tanh(
                            x_gates(n, d + 2 * H) + rh_proj(n, d) +
                            bias_(d + 2 * H));

                        // h_t = (1 - z) * n + z * h_prev
                        h_t(n, d) = (1.0f - z_gate(n, d)) * n_cand(n, d) +
                                    z_gate(n, d) * h_prev(n, d);

                        output(n, t, d) = h_t(n, d);
                    }
                }

                caches_[t].z_gate = z_gate;
                caches_[t].r_gate = r_gate;
                caches_[t].n_cand = n_cand;
                caches_[t].h_t = h_t;
                caches_[t].h_prev = h_prev;
                caches_[t].rh = rh;

                h_prev = h_t;
            }

            return output;
        }

        const Eigen::Tensor<float, 3> GRU::backward(
            const Eigen::Tensor<float, 3>& grad_output)
        {
            int batch = grad_output.dimension(0);
            int seq_len = grad_output.dimension(1);
            int H = hidden_size_;
            int gate3 = 3 * H;

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();

            Eigen::Tensor<float, 3> grad_input(batch, seq_len, input_size_);
            grad_input.setZero();

            #ifdef USE_CUDA
            if (device_ == "gpu") { backward_gpu(grad_output, grad_input, batch, seq_len); return grad_input; }
            #endif

            Eigen::Tensor<float, 2> dh_next(batch, H);
            dh_next.setZero();

            // extract W_hn sub-matrix for candidate computation
            Eigen::Tensor<float, 2> W_hn(hidden_size_, H);
            for (int i = 0; i < hidden_size_; ++i)
                for (int j = 0; j < H; ++j)
                    W_hn(i, j) = W_hh_(i, j + 2 * H);

            for (int t = seq_len - 1; t >= 0; --t)
            {
                auto& cache = caches_[t];

                Eigen::Tensor<float, 2> dh(batch, H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        dh(n, d) = grad_output(n, t, d) + dh_next(n, d);

                // backprop through h_t = (1-z)*n + z*h_prev
                Eigen::Tensor<float, 2> dn(batch, H);
                Eigen::Tensor<float, 2> dz(batch, H);
                Eigen::Tensor<float, 2> dh_prev(batch, H);

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                    {
                        dn(n, d) = dh(n, d) * (1.0f - cache.z_gate(n, d));
                        dz(n, d) = dh(n, d) * (cache.h_prev(n, d) - cache.n_cand(n, d));
                        dh_prev(n, d) = dh(n, d) * cache.z_gate(n, d);
                    }

                // dn through tanh: dn_raw = dn * (1 - n^2)
                Eigen::Tensor<float, 2> dn_raw(batch, H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        dn_raw(n, d) = dn(n, d) *
                            (1.0f - cache.n_cand(n, d) * cache.n_cand(n, d));

                // dz through sigmoid: dz_raw = dz * z * (1-z)
                Eigen::Tensor<float, 2> dz_raw(batch, H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        dz_raw(n, d) = dz(n, d) *
                            cache.z_gate(n, d) * (1.0f - cache.z_gate(n, d));

                Eigen::array<Eigen::IndexPair<int>, 1> contract_t = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> d_rh = dn_raw.contract(W_hn, contract_t);

                // d_r = d_rh * h_prev, dr through sigmoid
                Eigen::Tensor<float, 2> dr_raw(batch, H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                    {
                        float dr = d_rh(n, d) * cache.h_prev(n, d);
                        dr_raw(n, d) = dr * cache.r_gate(n, d) *
                                       (1.0f - cache.r_gate(n, d));
                        dh_prev(n, d) += d_rh(n, d) * cache.r_gate(n, d);
                    }

                // assemble dgates = [dz_raw | dr_raw | dn_raw]  [batch, 3*H]
                Eigen::Tensor<float, 2> dgates(batch, gate3);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                    {
                        dgates(n, d + 0 * H) = dz_raw(n, d);
                        dgates(n, d + 1 * H) = dr_raw(n, d);
                        dgates(n, d + 2 * H) = dn_raw(n, d);
                    }

                Eigen::Tensor<float, 2> x_t(batch, input_size_);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        x_t(n, d) = input_cache_(n, t, d);

                Eigen::array<Eigen::IndexPair<int>, 1> contract_batch = {Eigen::IndexPair<int>(0, 0)};
                grad_W_ih_ += x_t.contract(dgates, contract_batch);

                Eigen::Tensor<float, 2> h_for_grad(batch, gate3);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                    {
                        h_for_grad(n, d + 0 * H) = cache.h_prev(n, d);
                        h_for_grad(n, d + 1 * H) = cache.h_prev(n, d);
                        h_for_grad(n, d + 2 * H) = cache.rh(n, d);
                    }

                for (int n = 0; n < batch; ++n)
                {
                    for (int i = 0; i < hidden_size_; ++i)
                    {
                        for (int d = 0; d < H; ++d)
                        {
                            // z-gate
                            grad_W_hh_(i, d + 0 * H) += cache.h_prev(n, i) * dgates(n, d + 0 * H);
                            // r-gate
                            grad_W_hh_(i, d + 1 * H) += cache.h_prev(n, i) * dgates(n, d + 1 * H);
                            // n-candidate (uses rh)
                            grad_W_hh_(i, d + 2 * H) += cache.rh(n, i) * dgates(n, d + 2 * H);
                        }
                    }
                }

                for (int g = 0; g < gate3; ++g)
                    for (int n = 0; n < batch; ++n)
                        grad_bias_(g) += dgates(n, g);

                Eigen::Tensor<float, 2> dx = dgates.contract(W_ih_, contract_t);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < input_size_; ++d)
                        grad_input(n, t, d) = dx(n, d);

                Eigen::Tensor<float, 2> dgates_zr(batch, 2 * H);
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < 2 * H; ++d)
                        dgates_zr(n, d) = dgates(n, d);

                Eigen::Tensor<float, 2> W_hh_zr(hidden_size_, 2 * H);
                for (int i = 0; i < hidden_size_; ++i)
                    for (int j = 0; j < 2 * H; ++j)
                        W_hh_zr(i, j) = W_hh_(i, j);

                Eigen::array<Eigen::IndexPair<int>, 1> contract_2h = {Eigen::IndexPair<int>(1, 1)};
                Eigen::Tensor<float, 2> dh_from_gates = dgates_zr.contract(W_hh_zr, contract_2h);

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        dh_next(n, d) = dh_prev(n, d) + dh_from_gates(n, d);
            }

            return grad_input;
        }

        void GRU::step(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            if (!trainable_) return;

            optimizer.update(W_ih_.data(), grad_W_ih_.data(), W_ih_.size(), learning_rate);
            optimizer.update(W_hh_.data(), grad_W_hh_.data(), W_hh_.size(), learning_rate);
            optimizer.update(bias_.data(), grad_bias_.data(), bias_.size(), learning_rate);

            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        void GRU::reset_grads()
        {
            grad_W_ih_.setZero();
            grad_W_hh_.setZero();
            grad_bias_.setZero();
        }

        #ifdef USE_CUDA
        void GRU::forward_gpu(const Eigen::Tensor<float, 3>& input,
                              Eigen::Tensor<float, 3>& output,
                              int batch, int seq_len)
        {
            int I = input_size_, H = hidden_size_;
            int gate3 = 3 * H;

            float *d_W_ih, *d_W_hh, *d_bias;
            float *d_x_t, *d_h_prev;
            float *d_x_gates, *d_h_gates;
            float *d_z, *d_r, *d_rh;
            float *d_W_hn, *d_rh_proj;
            float *d_n, *d_h_t;

            cudaMalloc(&d_W_ih,     I * gate3 * sizeof(float));
            cudaMalloc(&d_W_hh,     H * gate3 * sizeof(float));
            cudaMalloc(&d_bias,     gate3 * sizeof(float));
            cudaMalloc(&d_x_t,      batch * I * sizeof(float));
            cudaMalloc(&d_h_prev,   batch * H * sizeof(float));
            cudaMalloc(&d_x_gates,  batch * gate3 * sizeof(float));
            cudaMalloc(&d_h_gates,  batch * gate3 * sizeof(float));
            cudaMalloc(&d_z,        batch * H * sizeof(float));
            cudaMalloc(&d_r,        batch * H * sizeof(float));
            cudaMalloc(&d_rh,       batch * H * sizeof(float));
            cudaMalloc(&d_W_hn,     H * H * sizeof(float));
            cudaMalloc(&d_rh_proj,  batch * H * sizeof(float));
            cudaMalloc(&d_n,        batch * H * sizeof(float));
            cudaMalloc(&d_h_t,      batch * H * sizeof(float));

            cudaMemcpy(d_W_ih, W_ih_.data(), I * gate3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_W_hh, W_hh_.data(), H * gate3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bias, bias_.data(), gate3 * sizeof(float),       cudaMemcpyHostToDevice);
            cudaMemset(d_h_prev, 0, batch * H * sizeof(float));

            // extract W_hn (columns 2H..3H of W_hh) on CPU then upload
            Eigen::Tensor<float, 2> W_hn(H, H);
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < H; ++j)
                    W_hn(i, j) = W_hh_(i, j + 2 * H);
            cudaMemcpy(d_W_hn, W_hn.data(), H * H * sizeof(float), cudaMemcpyHostToDevice);

            Eigen::Tensor<float, 2> x_t(batch, I);

            for (int t = 0; t < seq_len; ++t)
            {
                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < I; ++d)
                        x_t(n, d) = input(n, t, d);
                cudaMemcpy(d_x_t, x_t.data(), batch * I * sizeof(float), cudaMemcpyHostToDevice);

                // x_gates = x_t * W_ih  [batch, 3H]
                dim3 block(32, 32);
                dim3 grid((gate3 + 31) / 32, (batch + 31) / 32);
                Kernels::GPU::matmul_kernel<<<grid, block>>>(d_x_t, d_W_ih, d_x_gates, batch, gate3, I);

                // h_gates = h_prev * W_hh  [batch, 3H]
                Kernels::GPU::matmul_kernel<<<grid, block>>>(d_h_prev, d_W_hh, d_h_gates, batch, gate3, H);

                int total = batch * H;
                int bk = 256;
                int gr = (total + bk - 1) / bk;

                // z, r gates + rh = r * h_prev
                Kernels::GPU::gru_zr_forward_kernel<<<gr, bk>>>(
                    d_x_gates, d_h_gates, d_bias, d_h_prev,
                    d_z, d_r, d_rh, batch, H);

                // rh_proj = rh * W_hn  [batch, H]
                dim3 grid_hn((H + 31) / 32, (batch + 31) / 32);
                Kernels::GPU::matmul_kernel<<<grid_hn, block>>>(d_rh, d_W_hn, d_rh_proj, batch, H, H);

                // n = tanh(x_gates_n + rh_proj + b_n), h_t = (1-z)*n + z*h_prev
                Kernels::GPU::gru_output_forward_kernel<<<gr, bk>>>(
                    d_x_gates, d_rh_proj, d_bias, d_z, d_h_prev,
                    d_n, d_h_t, batch, H);

                // copy caches to CPU
                Eigen::Tensor<float, 2> buf(batch, H);
                cudaMemcpy(buf.data(), d_z, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].z_gate = buf;
                cudaMemcpy(buf.data(), d_r, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].r_gate = buf;
                cudaMemcpy(buf.data(), d_n, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].n_cand = buf;
                cudaMemcpy(buf.data(), d_h_t, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].h_t = buf;
                cudaMemcpy(buf.data(), d_rh, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].rh = buf;

                Eigen::Tensor<float, 2> hp(batch, H);
                cudaMemcpy(hp.data(), d_h_prev, batch * H * sizeof(float), cudaMemcpyDeviceToHost);
                caches_[t].h_prev = hp;

                for (int n = 0; n < batch; ++n)
                    for (int d = 0; d < H; ++d)
                        output(n, t, d) = caches_[t].h_t(n, d);

                cudaMemcpy(d_h_prev, d_h_t, batch * H * sizeof(float), cudaMemcpyDeviceToDevice);
            }

            cudaFree(d_W_ih); cudaFree(d_W_hh); cudaFree(d_bias);
            cudaFree(d_x_t); cudaFree(d_h_prev);
            cudaFree(d_x_gates); cudaFree(d_h_gates);
            cudaFree(d_z); cudaFree(d_r); cudaFree(d_rh);
            cudaFree(d_W_hn); cudaFree(d_rh_proj);
            cudaFree(d_n); cudaFree(d_h_t);
        }

        void GRU::backward_gpu(const Eigen::Tensor<float, 3>& grad_output,
                               Eigen::Tensor<float, 3>& grad_input,
                               int batch, int seq_len)
        {
            int I = input_size_, H = hidden_size_;
            int gate3 = 3 * H;

            float *d_W_ih, *d_W_hh, *d_W_hn;
            float *d_x_t, *d_h_prev, *d_rh_cache;
            float *d_z, *d_r, *d_n, *d_dh, *d_dh_next;
            float *d_dn_raw, *d_dz_raw, *d_dr_raw, *d_dh_prev_z;
            float *d_d_rh, *d_dh_prev_complete;
            float *d_dgates;
            float *d_dW_ih, *d_dW_hh, *d_dbias;
            float *d_dW_ih_t, *d_dW_hh_zr_t, *d_dW_hn_t;
            float *d_dx;
            float *d_dgates_zr, *d_W_hh_zr, *d_dh_from_gates;

            cudaMalloc(&d_W_ih,              I * gate3 * sizeof(float));
            cudaMalloc(&d_W_hh,              H * gate3 * sizeof(float));
            cudaMalloc(&d_W_hn,              H * H * sizeof(float));
            cudaMalloc(&d_x_t,               batch * I * sizeof(float));
            cudaMalloc(&d_h_prev,            batch * H * sizeof(float));
            cudaMalloc(&d_rh_cache,          batch * H * sizeof(float));
            cudaMalloc(&d_z,                 batch * H * sizeof(float));
            cudaMalloc(&d_r,                 batch * H * sizeof(float));
            cudaMalloc(&d_n,                 batch * H * sizeof(float));
            cudaMalloc(&d_dh,                batch * H * sizeof(float));
            cudaMalloc(&d_dh_next,           batch * H * sizeof(float));
            cudaMalloc(&d_dn_raw,            batch * H * sizeof(float));
            cudaMalloc(&d_dz_raw,            batch * H * sizeof(float));
            cudaMalloc(&d_dr_raw,            batch * H * sizeof(float));
            cudaMalloc(&d_dh_prev_z,         batch * H * sizeof(float));
            cudaMalloc(&d_d_rh,              batch * H * sizeof(float));
            cudaMalloc(&d_dh_prev_complete,  batch * H * sizeof(float));
            cudaMalloc(&d_dgates,            batch * gate3 * sizeof(float));
            cudaMalloc(&d_dW_ih,             I * gate3 * sizeof(float));
            cudaMalloc(&d_dW_hh,             H * gate3 * sizeof(float));
            cudaMalloc(&d_dbias,             gate3 * sizeof(float));
            cudaMalloc(&d_dW_ih_t,           I * gate3 * sizeof(float));
            cudaMalloc(&d_dW_hh_zr_t,        H * 2 * H * sizeof(float));
            cudaMalloc(&d_dW_hn_t,           H * H * sizeof(float));
            cudaMalloc(&d_dx,                batch * I * sizeof(float));
            cudaMalloc(&d_dgates_zr,         batch * 2 * H * sizeof(float));
            cudaMalloc(&d_W_hh_zr,           H * 2 * H * sizeof(float));
            cudaMalloc(&d_dh_from_gates,     batch * H * sizeof(float));

            cudaMemcpy(d_W_ih, W_ih_.data(), I * gate3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_W_hh, W_hh_.data(), H * gate3 * sizeof(float), cudaMemcpyHostToDevice);

            // extract W_hn and W_hh_zr
            Eigen::Tensor<float, 2> W_hn(H, H);
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < H; ++j)
                    W_hn(i, j) = W_hh_(i, j + 2 * H);
            cudaMemcpy(d_W_hn, W_hn.data(), H * H * sizeof(float), cudaMemcpyHostToDevice);

            Eigen::Tensor<float, 2> W_hh_zr(H, 2 * H);
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < 2 * H; ++j)
                    W_hh_zr(i, j) = W_hh_(i, j);
            cudaMemcpy(d_W_hh_zr, W_hh_zr.data(), H * 2 * H * sizeof(float), cudaMemcpyHostToDevice);

            cudaMemset(d_dW_ih,   0, I * gate3 * sizeof(float));
            cudaMemset(d_dW_hh,   0, H * gate3 * sizeof(float));
            cudaMemset(d_dbias,   0, gate3 * sizeof(float));
            cudaMemset(d_dh_next, 0, batch * H * sizeof(float));

            Eigen::Tensor<float, 2> x_t(batch, I);
            Eigen::Tensor<float, 2> grad_t(batch, H);

            for (int t = seq_len - 1; t >= 0; --t)
            {
                auto& cache = caches_[t];

                // upload grad slice
                for (int nn = 0; nn < batch; ++nn)
                    for (int d = 0; d < H; ++d)
                        grad_t(nn, d) = grad_output(nn, t, d);
                cudaMemcpy(d_dh, grad_t.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);

                // upload cached gate values
                cudaMemcpy(d_z, cache.z_gate.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_r, cache.r_gate.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_n, cache.n_cand.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_h_prev, cache.h_prev.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_rh_cache, cache.rh.data(), batch * H * sizeof(float), cudaMemcpyHostToDevice);

                int total = batch * H;
                int bk = 256;
                int gr = (total + bk - 1) / bk;

                Kernels::GPU::gru_bwd_gates_kernel<<<gr, bk>>>(
                    d_dh, d_dh_next, d_z, d_n, d_h_prev,
                    d_dn_raw, d_dz_raw, d_dh_prev_z, total);

                dim3 block(32, 32);
                dim3 grid_hn((H + 31) / 32, (batch + 31) / 32);
                cudaMemset(d_d_rh, 0, batch * H * sizeof(float));
                Kernels::GPU::matmul_grad_input_kernel<<<grid_hn, block>>>(
                    d_dn_raw, d_W_hn, d_d_rh, batch, H, H);

                Kernels::GPU::gru_bwd_reset_kernel<<<gr, bk>>>(
                    d_d_rh, d_h_prev, d_r, d_dh_prev_z,
                    d_dr_raw, d_dh_prev_complete, total);

                Kernels::GPU::gru_assemble_dgates_kernel<<<gr, bk>>>(
                    d_dz_raw, d_dr_raw, d_dn_raw, d_dgates, batch, H);

                for (int nn = 0; nn < batch; ++nn)
                    for (int d = 0; d < I; ++d)
                        x_t(nn, d) = input_cache_(nn, t, d);
                cudaMemcpy(d_x_t, x_t.data(), batch * I * sizeof(float), cudaMemcpyHostToDevice);

                // grad_W_ih += x_t^T * dgates
                int ws = I * gate3;
                cudaMemset(d_dW_ih_t, 0, ws * sizeof(float));
                dim3 grid_wih((gate3 + 31) / 32, (I + 31) / 32);
                Kernels::GPU::matmul_grad_weights_kernel<<<grid_wih, block>>>(
                    d_x_t, d_dgates, d_dW_ih_t, batch, I, gate3);
                Kernels::GPU::elementwise_kernel<<<(ws + bk - 1) / bk, bk>>>(
                    d_dW_ih, d_dW_ih_t, d_dW_ih, ws, 0, 0);

                cudaMemcpy(d_dgates_zr, d_dgates, batch * 2 * H * sizeof(float), cudaMemcpyDeviceToDevice);

                int wzr = H * 2 * H;
                cudaMemset(d_dW_hh_zr_t, 0, wzr * sizeof(float));
                dim3 grid_whzr((2 * H + 31) / 32, (H + 31) / 32);
                Kernels::GPU::matmul_grad_weights_kernel<<<grid_whzr, block>>>(
                    d_h_prev, d_dgates_zr, d_dW_hh_zr_t, batch, H, 2 * H);
                Kernels::GPU::elementwise_kernel<<<(wzr + bk - 1) / bk, bk>>>(
                    d_dW_hh, d_dW_hh_zr_t, d_dW_hh, wzr, 0, 0);

                int wn = H * H;
                cudaMemset(d_dW_hn_t, 0, wn * sizeof(float));
                dim3 grid_whn((H + 31) / 32, (H + 31) / 32);
                Kernels::GPU::matmul_grad_weights_kernel<<<grid_whn, block>>>(
                    d_rh_cache, d_dn_raw, d_dW_hn_t, batch, H, H);
                float* d_dW_hh_n_ptr = d_dW_hh + H * 2 * H;
                Kernels::GPU::elementwise_kernel<<<(wn + bk - 1) / bk, bk>>>(
                    d_dW_hh_n_ptr, d_dW_hn_t, d_dW_hh_n_ptr, wn, 0, 0);

                Kernels::GPU::bias_grad_kernel<<<gate3, 256>>>(d_dgates, d_dbias, batch, gate3);

                // grad_input = dgates * W_ih^T
                dim3 grid_dx((I + 31) / 32, (batch + 31) / 32);
                cudaMemset(d_dx, 0, batch * I * sizeof(float));
                Kernels::GPU::matmul_grad_input_kernel<<<grid_dx, block>>>(
                    d_dgates, d_W_ih, d_dx, batch, I, gate3);
                Eigen::Tensor<float, 2> dx(batch, I);
                cudaMemcpy(dx.data(), d_dx, batch * I * sizeof(float), cudaMemcpyDeviceToHost);
                for (int nn = 0; nn < batch; ++nn)
                    for (int d = 0; d < I; ++d)
                        grad_input(nn, t, d) = dx(nn, d);

                // dh_next = dh_prev_complete + dgates_zr * W_hh_zr^T
                cudaMemset(d_dh_from_gates, 0, batch * H * sizeof(float));
                Kernels::GPU::matmul_grad_input_kernel<<<grid_hn, block>>>(
                    d_dgates_zr, d_W_hh_zr, d_dh_from_gates, batch, H, 2 * H);
                Kernels::GPU::elementwise_kernel<<<(total + bk - 1) / bk, bk>>>(
                    d_dh_prev_complete, d_dh_from_gates, d_dh_next, total, 0, 0);
            }

            cudaMemcpy(grad_W_ih_.data(), d_dW_ih, I * gate3 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_W_hh_.data(), d_dW_hh, H * gate3 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_bias_.data(), d_dbias, gate3 * sizeof(float),       cudaMemcpyDeviceToHost);

            cudaFree(d_W_ih); cudaFree(d_W_hh); cudaFree(d_W_hn);
            cudaFree(d_x_t); cudaFree(d_h_prev); cudaFree(d_rh_cache);
            cudaFree(d_z); cudaFree(d_r); cudaFree(d_n);
            cudaFree(d_dh); cudaFree(d_dh_next);
            cudaFree(d_dn_raw); cudaFree(d_dz_raw); cudaFree(d_dr_raw);
            cudaFree(d_dh_prev_z); cudaFree(d_d_rh); cudaFree(d_dh_prev_complete);
            cudaFree(d_dgates);
            cudaFree(d_dW_ih); cudaFree(d_dW_hh); cudaFree(d_dbias);
            cudaFree(d_dW_ih_t); cudaFree(d_dW_hh_zr_t); cudaFree(d_dW_hn_t);
            cudaFree(d_dx);
            cudaFree(d_dgates_zr); cudaFree(d_W_hh_zr); cudaFree(d_dh_from_gates);
        }
        #endif // USE_CUDA
    }
}
