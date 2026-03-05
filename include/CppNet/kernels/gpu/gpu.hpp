#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>


namespace CppNet 
{
    namespace Kernels 
    {
        namespace GPU 
        {

            __global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K);

            __global__ void add_bias_kernel(float* output, const float* bias, int M, int N);

            __global__ void matmul_grad_weights_kernel(const float* X, const float* dY, float* dW, int batch, int in_size, int out_size);

            __global__ void bias_grad_kernel(const float* dY, float* db, int batch, int out);

            __global__ void matmul_grad_input_kernel(const float* dY, const float* W, float* dX, int batch, int in_size, int out_size);

            __global__ void elementwise_kernel(float* A, float* B, float* C, int N, int op, int unused);

            __global__ void sgd_step_kernel(float* W, const float* dW, float LR, int TP);

           __global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int total_elements);

            __global__ void relu_grad_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int total_elements);

            __global__ void leaky_relu_kernel(const float* __restrict__ input, float* __restrict__ output, int total_elements, float alpha);

            __global__ void leaky_relu_grad_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int total_elements, float alpha);

            __global__ void tanh_kernel(const float* __restrict__ input, float* __restrict__ output, int total_elements);

            __global__ void tanh_grad_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int total_elements);

            __global__ void sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int total_elements);

            __global__ void sigmoid_grad_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int total_elements);

            __global__ void conv2d_forward_kernel(
                const float* input, const float* weights, const float* bias,
                float* output,
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride, int padding,
                int H_out, int W_out,
                bool use_bias);

            __global__ void conv2d_backward_kernel(
                const float* grad_output, const float* input_cache,
                const float* weights,
                float* grad_input, float* grad_weights, float* grad_biases,
                int N, int C_in, int H, int W,
                int C_out, int kH, int kW,
                int stride, int padding,
                int H_out, int W_out,
                bool use_bias);

            __global__ void maxpool2d_forward_kernel(
                const float* input, float* output, int* max_indices,
                int N, int C, int H, int W,
                int pool_size, int stride,
                int H_out, int W_out);

            __global__ void maxpool2d_backward_kernel(
                const float* grad_output, const int* max_indices,
                float* grad_input,
                int N, int C, int H, int W,
                int pool_size, int stride,
                int H_out, int W_out);

            __global__ void rnn_tanh_forward_kernel(
                const float* pre_ih, const float* pre_hh,
                const float* bias, float* h_t,
                int total, int hidden);

            __global__ void rnn_tanh_backward_kernel(
                const float* grad, const float* dh_next,
                const float* h_t, float* dtanh,
                int total, int hidden);

            __global__ void lstm_gates_forward_kernel(
                const float* pre_ih, const float* pre_hh,
                const float* bias, const float* c_prev,
                float* i_gate, float* f_gate,
                float* g_gate, float* o_gate,
                float* c_t, float* h_t, float* tanh_c,
                int batch, int hidden);

            __global__ void lstm_gates_backward_kernel(
                const float* dh, const float* dc_next,
                const float* i_gate, const float* f_gate,
                const float* g_gate, const float* o_gate,
                const float* tanh_c, const float* c_prev,
                float* dgates, float* dc_out,
                int batch, int hidden);

            __global__ void gru_zr_forward_kernel(
                const float* x_gates, const float* h_gates,
                const float* bias, const float* h_prev,
                float* z_gate, float* r_gate, float* rh,
                int batch, int hidden);

            __global__ void gru_output_forward_kernel(
                const float* x_gates, const float* rh_proj,
                const float* bias, const float* z_gate,
                const float* h_prev,
                float* n_cand, float* h_t,
                int batch, int hidden);

            __global__ void gru_bwd_gates_kernel(
                const float* dh_raw, const float* dh_next_in,
                const float* z_gate, const float* n_cand,
                const float* h_prev,
                float* dn_raw, float* dz_raw, float* dh_prev_z,
                int total);

            __global__ void gru_bwd_reset_kernel(
                const float* d_rh, const float* h_prev,
                const float* r_gate, const float* dh_prev_z,
                float* dr_raw, float* dh_prev_out,
                int total);

            __global__ void gru_assemble_dgates_kernel(
                const float* dz_raw, const float* dr_raw,
                const float* dn_raw, float* dgates,
                int batch, int hidden);

            void elementwise_gpu(const float* A, const float* B, float* C, int N, int op);

            __global__ void embedding_forward_kernel(
                const int* input, const float* weight, float* output,
                int batch, int seq_len, int embed_dim, int vocab_size);

            __global__ void embedding_backward_kernel(
                const int* input, const float* grad_output,
                float* grad_weight,
                int batch, int seq_len, int embed_dim, int vocab_size);

            __global__ void attention_scale_kernel(
                float* scores, float scale, int total);

            __global__ void attention_softmax_forward_kernel(
                const float* scores, float* output,
                int rows, int cols);

            __global__ void attention_softmax_backward_kernel(
                const float* grad_attn, const float* attn,
                float* grad_scores,
                int rows, int cols);

            __global__ void mean_pool1d_forward_kernel(
                const float* input, float* output,
                int B, int S, int D);

            __global__ void mean_pool1d_backward_kernel(
                const float* grad_output, float* grad_input,
                int B, int S, int D);

            __global__ void batch_norm_forward_kernel(
                const float* input, float* output, float* x_hat,
                const float* gamma, const float* beta,
                float* batch_mean, float* batch_var,
                float* running_mean, float* running_var,
                int batch, int features,
                float eps, float momentum, bool training);

            __global__ void batch_norm_backward_kernel(
                const float* grad_output, const float* input,
                const float* x_hat, const float* gamma,
                const float* batch_mean, const float* batch_var,
                float* grad_input, float* grad_gamma, float* grad_beta,
                int batch, int features, float eps);

            __global__ void dropout_forward_kernel(
                const float* input, const float* mask,
                float* output, int total, float scale);

            __global__ void dropout_backward_kernel(
                const float* grad_output, const float* mask,
                float* grad_input, int total, float scale);

            __global__ void global_avg_pool2d_forward_kernel(
                const float* input, float* output,
                int batch, int channels, int height, int width);

            __global__ void global_avg_pool2d_backward_kernel(
                const float* grad_output, float* grad_input,
                int batch, int channels, int height, int width);

            __global__ void global_max_pool2d_forward_kernel(
                const float* input, float* output, int* argmax,
                int batch, int channels, int height, int width);

            __global__ void global_max_pool2d_backward_kernel(
                const float* grad_output, const int* argmax,
                float* grad_input,
                int batch, int channels, int spatial);

            // ---- Loss kernels ----

            __global__ void mse_forward_kernel(
                const float* pred, const float* target, float* loss, int total);

            __global__ void mse_backward_kernel(
                const float* pred, const float* target, float* grad,
                int total, float scale);

            __global__ void mae_forward_kernel(
                const float* pred, const float* target, float* loss, int total);

            __global__ void mae_backward_kernel(
                const float* pred, const float* target, float* grad,
                int total, float scale);

            __global__ void huber_forward_kernel(
                const float* pred, const float* target, float* loss,
                int total, float delta);

            __global__ void huber_backward_kernel(
                const float* pred, const float* target, float* grad,
                int total, float scale, float delta);

            __global__ void bce_forward_kernel(
                const float* pred, const float* target, float* loss, int total);

            __global__ void bce_backward_kernel(
                const float* pred, const float* target, float* grad,
                int total, float scale);

            __global__ void softmax_ce_forward_kernel(
                const float* logits, const float* targets,
                float* softmax_out, float* loss,
                int batch, int classes);

            __global__ void softmax_ce_backward_kernel(
                const float* softmax, const float* targets, float* grad,
                int total, float scale);

            __global__ void categorical_ce_logits_forward_kernel(
                const float* logits, const float* targets,
                float* softmax_out, float* loss,
                int batch, int classes);

            __global__ void categorical_ce_logits_backward_kernel(
                const float* softmax, const float* targets, float* grad,
                int total, float scale);

            __global__ void categorical_ce_probs_forward_kernel(
                const float* pred, const float* targets, float* loss, int total);

            __global__ void categorical_ce_probs_backward_kernel(
                const float* pred, const float* targets, float* grad,
                int total, float scale);

        }
    }
}

#endif // USE_CUDA