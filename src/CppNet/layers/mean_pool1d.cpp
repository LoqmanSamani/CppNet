/**
 * @file mean_pool1d.cpp
 * @brief MeanPool1D implementation — averages over the sequence dimension
 *
 * Forward:  output[b,d] = (1/S) * sum_s input[b,s,d]
 * Backward: grad_input[b,s,d] = grad_output[b,d] / S
 *
 * Supports three backends: cpu-eigen, cpu (OpenMP), gpu (CUDA).
 */

#include "CppNet/layers/mean_pool1d.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Layers
    {
        MeanPool1D::MeanPool1D(const std::string& device)
            : device_(device) {}

        Eigen::Tensor<float, 2> MeanPool1D::forward(
            const Eigen::Tensor<float, 3>& input)
        {
            batch_cache_    = input.dimension(0);
            seq_len_cache_  = input.dimension(1);
            features_cache_ = input.dimension(2);

            int B = batch_cache_;
            int S = seq_len_cache_;
            int D = features_cache_;

            Eigen::Tensor<float, 2> output(B, D);

            if (device_ == "cpu-eigen")
            {
                // Eigen: mean over dimension 1
                Eigen::array<int, 1> reduce_dim = {1};
                output = input.mean(reduce_dim);
            }
#ifdef USE_CUDA
            else if (device_ == "gpu")
            {
                int total_bd = B * D;

                float* d_input  = nullptr;
                float* d_output = nullptr;

                cudaMalloc(&d_input,  B * S * D * sizeof(float));
                cudaMalloc(&d_output, B * D * sizeof(float));

                cudaMemcpy(d_input, input.data(),
                           B * S * D * sizeof(float), cudaMemcpyHostToDevice);

                int threads = 256;
                int blocks  = (total_bd + threads - 1) / threads;
                Kernels::GPU::mean_pool1d_forward_kernel<<<blocks, threads>>>(
                    d_input, d_output, B, S, D);

                cudaMemcpy(output.data(), d_output,
                           B * D * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_input);
                cudaFree(d_output);
            }
#endif
            else // "cpu" — OpenMP
            {
                float inv_S = 1.0f / static_cast<float>(S);
                int BD = B * D;
                output.setZero();

                #pragma omp parallel for schedule(static)
                for (int bd = 0; bd < BD; ++bd)
                {
                    int b = bd / D;
                    int d = bd % D;
                    float sum = 0.0f;
                    for (int s = 0; s < S; ++s)
                        sum += input(b, s, d);
                    output(b, d) = sum * inv_S;
                }
            }

            return output;
        }

        Eigen::Tensor<float, 3> MeanPool1D::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            int B = batch_cache_;
            int S = seq_len_cache_;
            int D = features_cache_;

            Eigen::Tensor<float, 3> grad_input(B, S, D);

            if (device_ == "cpu-eigen")
            {
                float inv_S = 1.0f / static_cast<float>(S);
                // broadcast grad_output to [B, S, D]
                for (int b = 0; b < B; ++b)
                    for (int s = 0; s < S; ++s)
                        for (int d = 0; d < D; ++d)
                            grad_input(b, s, d) = grad_output(b, d) * inv_S;
            }
#ifdef USE_CUDA
            else if (device_ == "gpu")
            {
                int total_bsd = B * S * D;

                float* d_grad_output = nullptr;
                float* d_grad_input  = nullptr;

                cudaMalloc(&d_grad_output, B * D * sizeof(float));
                cudaMalloc(&d_grad_input,  total_bsd * sizeof(float));

                cudaMemcpy(d_grad_output, grad_output.data(),
                           B * D * sizeof(float), cudaMemcpyHostToDevice);

                int threads = 256;
                int blocks  = (total_bsd + threads - 1) / threads;
                Kernels::GPU::mean_pool1d_backward_kernel<<<blocks, threads>>>(
                    d_grad_output, d_grad_input, B, S, D);

                cudaMemcpy(grad_input.data(), d_grad_input,
                           total_bsd * sizeof(float), cudaMemcpyDeviceToHost);

                cudaFree(d_grad_output);
                cudaFree(d_grad_input);
            }
#endif
            else // "cpu" — OpenMP
            {
                float inv_S = 1.0f / static_cast<float>(S);
                int total = B * S * D;

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < total; ++i)
                {
                    int d = i / (B * S);
                    int rem = i % (B * S);
                    int s = rem / B;
                    int b = rem % B;
                    grad_input(b, s, d) = grad_output(b, d) * inv_S;
                }
            }

            return grad_input;
        }

        void MeanPool1D::step(Optimizers::Optimizer&, float) {}
    }
}
