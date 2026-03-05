/**
 * @file gru_sequence_prediction.cpp
 * @brief GRU-based recurrent network for sine-wave regression.
 *
 * Architecture:
 *   GRU(1, 16, return_sequences=true) -> take last timestep -> Linear(16, 1)
 *
 * Loss:     MAE (Mean Absolute Error)
 * Optimizer: Momentum
 * Data:     Predict next value in a discretised sine wave.
 *           Input: [sin(t), sin(t+1), ..., sin(t+seq-1)]
 *           Target: sin(t+seq)
 *
 * Demonstrates: GRU layer, MAE loss, Momentum optimizer
 */

#include <CppNet/CppNet.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace CppNet;

static void generate_sine_data(
    Eigen::Tensor<float, 3>& X,
    Eigen::Tensor<float, 2>& Y,
    int num_samples,
    int seq_len)
{
    X.resize(num_samples, seq_len, 1);
    Y.resize(num_samples, 1);
    const float step = 0.2f;
    for (int i = 0; i < num_samples; ++i)
    {
        float start = i * step;
        for (int t = 0; t < seq_len; ++t)
            X(i, t, 0) = std::sin(start + t * step);
        Y(i, 0) = std::sin(start + seq_len * step);
    }
}

int main()
{
    std::cout << "=== GRU Sine-Wave Prediction ===\n\n";

    const int   seq_len       = 5;
    const int   num_samples   = 400;
    const int   batch_size    = 32;
    const int   epochs        = 150;
    const float learning_rate = 0.01f;
    const int   hidden_size   = 16;

    Eigen::Tensor<float, 3> X;
    Eigen::Tensor<float, 2> Y;
    generate_sine_data(X, Y, num_samples, seq_len);
    std::cout << "Dataset: " << num_samples << " sequences, length " << seq_len
              << ", predicting next value\n";

    auto gru    = std::make_shared<Layers::GRU>(1, hidden_size, true, "cpu-eigen");
    auto linear = std::make_shared<Layers::Linear>(hidden_size, 1, "output",
                                                    true, true, "cpu-eigen");

    Models::SequentialModel model;
    model.add_layer(gru);
    model.add_layer(linear);

    Losses::MAE loss;
    Optimizers::Momentum optimizer(0.9f);

    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    model.summary();
    std::cout << "\n";

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0.0f;
        int   num_batches = 0;

        for (int start = 0; start + batch_size <= num_samples; start += batch_size)
        {
            Eigen::Tensor<float, 3> x_batch(batch_size, seq_len, 1);
            Eigen::Tensor<float, 2> y_batch(batch_size, 1);
            for (int i = 0; i < batch_size; ++i)
            {
                int idx = indices[start + i];
                for (int t = 0; t < seq_len; ++t)
                    x_batch(i, t, 0) = X(idx, t, 0);
                y_batch(i, 0) = Y(idx, 0);
            }

            auto gru_out = gru->forward(x_batch);

            // Take last timestep
            Eigen::Tensor<float, 2> last_hidden(batch_size, hidden_size);
            for (int b = 0; b < batch_size; ++b)
                for (int d = 0; d < hidden_size; ++d)
                    last_hidden(b, d) = gru_out(b, seq_len - 1, d);

            auto pred = linear->forward(last_hidden);
            float batch_loss = loss.forward(pred, y_batch);
            epoch_loss += batch_loss;
            ++num_batches;

            auto grad_pred = loss.backward(pred, y_batch);
            auto grad_hidden = linear->backward(grad_pred);

            Eigen::Tensor<float, 3> grad_gru(batch_size, seq_len, hidden_size);
            grad_gru.setZero();
            for (int b = 0; b < batch_size; ++b)
                for (int d = 0; d < hidden_size; ++d)
                    grad_gru(b, seq_len - 1, d) = grad_hidden(b, d);

            gru->backward(grad_gru);
            model.update(optimizer, learning_rate);
            linear->reset_grads();
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0)
        {
            std::cout << std::fixed << std::setprecision(6)
                      << "Epoch " << std::setw(3) << epoch + 1
                      << "  |  MAE Loss: " << epoch_loss / num_batches << "\n";
        }
    }

    // Evaluate on 10 samples
    const int eval_count = 10;
    std::cout << "\nSample predictions:\n";
    std::cout << std::setw(12) << "Predicted" << std::setw(12) << "Actual" << "\n";
    std::cout << std::string(24, '-') << "\n";

    Eigen::Tensor<float, 3> x_eval(eval_count, seq_len, 1);
    Eigen::Tensor<float, 2> y_eval(eval_count, 1);
    for (int i = 0; i < eval_count; ++i)
    {
        int idx = i * (num_samples / eval_count);
        for (int t = 0; t < seq_len; ++t)
            x_eval(i, t, 0) = X(idx, t, 0);
        y_eval(i, 0) = Y(idx, 0);
    }

    auto gru_out = gru->forward(x_eval);
    Eigen::Tensor<float, 2> last_h(eval_count, hidden_size);
    for (int b = 0; b < eval_count; ++b)
        for (int d = 0; d < hidden_size; ++d)
            last_h(b, d) = gru_out(b, seq_len - 1, d);
    auto preds = linear->forward(last_h);

    float mae_eval = 0.0f;
    for (int i = 0; i < eval_count; ++i)
    {
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(12) << preds(i, 0)
                  << std::setw(12) << y_eval(i, 0) << "\n";
        mae_eval += std::fabs(preds(i, 0) - y_eval(i, 0));
    }
    std::cout << "\nEval MAE: " << std::fixed << std::setprecision(6)
              << mae_eval / eval_count << "\n";

    std::cout << "\n=== GRU Example Complete ===\n";
    return 0;
}
