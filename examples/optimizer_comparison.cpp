/**
 * @file optimizer_comparison.cpp
 * @brief Compares five optimizers (SGD, Momentum, Adagrad, RMSProp, Adam)
 *        on the same MLP and dataset.
 *
 * Architecture:
 *   Linear(2, 32) -> Tanh -> Linear(32, 16) -> Tanh -> Linear(16, 1)
 *
 * Loss:     Huber (smooth L1)
 * Data:     Noisy sine regression: y = sin(x0) * cos(x1) + noise
 *
 * Demonstrates: SGD, Momentum, Adagrad, RMSProp (+ Adam baseline),
 *               Tanh activation, Huber loss
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

static void generate_regression_data(
    Eigen::Tensor<float, 2>& X,
    Eigen::Tensor<float, 2>& Y,
    int N,
    float noise = 0.05f)
{
    X.resize(N, 2);
    Y.resize(N, 1);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::normal_distribution<float> nd(0.0f, noise);
    for (int i = 0; i < N; ++i)
    {
        float x0 = dist(rng);
        float x1 = dist(rng);
        X(i, 0) = x0;
        X(i, 1) = x1;
        Y(i, 0) = std::sin(x0) * std::cos(x1) + nd(rng);
    }
}

// Train one model and return per-epoch losses
static std::vector<float> train_with_optimizer(
    const Eigen::Tensor<float, 2>& X,
    const Eigen::Tensor<float, 2>& Y,
    Optimizers::Optimizer& optimizer,
    const std::string& opt_name,
    int epochs,
    int batch_size,
    float learning_rate)
{
    const int N = X.dimension(0);

    auto linear1 = std::make_shared<Layers::Linear>(2, 32, "fc1", true, true, "cpu-eigen", "xavier");
    auto linear2 = std::make_shared<Layers::Linear>(32, 16, "fc2", true, true, "cpu-eigen", "xavier");
    auto linear3 = std::make_shared<Layers::Linear>(16, 1, "fc3", true, true, "cpu-eigen", "xavier");

    Activations::Tanh tanh1;
    Activations::Tanh tanh2;

    Models::SequentialModel model;
    model.add_layer(linear1);
    model.add_layer(linear2);
    model.add_layer(linear3);

    Losses::Huber loss(1.0f);

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123);

    std::vector<float> epoch_losses;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        float total_loss = 0.0f;
        int   num_batches = 0;

        for (int start = 0; start + batch_size <= N; start += batch_size)
        {
            Eigen::Tensor<float, 2> x_batch(batch_size, 2);
            Eigen::Tensor<float, 2> y_batch(batch_size, 1);
            for (int i = 0; i < batch_size; ++i)
            {
                int idx = indices[start + i];
                x_batch(i, 0) = X(idx, 0);
                x_batch(i, 1) = X(idx, 1);
                y_batch(i, 0) = Y(idx, 0);
            }

            // Forward
            auto z1 = linear1->forward(x_batch);
            auto a1 = tanh1.forward(z1);
            auto z2 = linear2->forward(a1);
            auto a2 = tanh2.forward(z2);
            auto z3 = linear3->forward(a2);

            float batch_loss = loss.forward(z3, y_batch);
            total_loss += batch_loss;
            ++num_batches;

            // Backward
            auto grad = loss.backward(z3, y_batch);
            grad = linear3->backward(grad);
            grad = tanh2.backward(grad);
            grad = linear2->backward(grad);
            grad = tanh1.backward(grad);
            linear1->backward(grad);

            // Update
            model.update(optimizer, learning_rate);
            linear1->reset_grads();
            linear2->reset_grads();
            linear3->reset_grads();
        }

        epoch_losses.push_back(total_loss / num_batches);
    }

    return epoch_losses;
}

int main()
{
    std::cout << "=== Optimizer Comparison (SGD, Momentum, Adagrad, RMSProp, Adam) ===\n\n";

    const int   N             = 500;
    const int   batch_size    = 32;
    const int   epochs        = 50;
    const float learning_rate = 0.01f;

    Eigen::Tensor<float, 2> X, Y;
    generate_regression_data(X, Y, N);
    std::cout << "Dataset: " << N << " samples, regression y = sin(x0)*cos(x1)\n";
    std::cout << "Loss: Huber | Activation: Tanh | Epochs: " << epochs << "\n\n";

    // Create optimizers
    Optimizers::SGD      opt_sgd;
    Optimizers::Momentum opt_momentum(0.9f);
    Optimizers::Adagrad  opt_adagrad;
    Optimizers::RMSProp  opt_rmsprop;
    Optimizers::Adam     opt_adam;

    struct OptimizerEntry {
        std::string name;
        Optimizers::Optimizer* opt;
    };

    OptimizerEntry entries[] = {
        {"SGD",      &opt_sgd},
        {"Momentum", &opt_momentum},
        {"Adagrad",  &opt_adagrad},
        {"RMSProp",  &opt_rmsprop},
        {"Adam",     &opt_adam},
    };

    // Collect results
    std::vector<std::pair<std::string, std::vector<float>>> results;
    for (auto& e : entries)
    {
        std::cout << "Training with " << e.name << "..." << std::flush;
        auto losses = train_with_optimizer(X, Y, *e.opt, e.name,
                                           epochs, batch_size, learning_rate);
        results.push_back({e.name, losses});
        std::cout << " done (final loss: " << std::fixed << std::setprecision(6)
                  << losses.back() << ")\n";
    }

    // Print comparison table at key epochs
    std::cout << "\n" << std::setw(8) << "Epoch";
    for (auto& r : results)
        std::cout << std::setw(12) << r.first;
    std::cout << "\n" << std::string(8 + 12 * results.size(), '-') << "\n";

    int checkpoints[] = {1, 5, 10, 20, 30, 40, 50};
    for (int ep : checkpoints)
    {
        if (ep > epochs) break;
        std::cout << std::setw(8) << ep;
        for (auto& r : results)
            std::cout << std::setw(12) << std::fixed << std::setprecision(6)
                      << r.second[ep - 1];
        std::cout << "\n";
    }

    // Final ranking
    std::cout << "\nFinal ranking (lowest loss first):\n";
    std::vector<std::pair<float, std::string>> ranking;
    for (auto& r : results)
        ranking.push_back({r.second.back(), r.first});
    std::sort(ranking.begin(), ranking.end());
    for (int i = 0; i < static_cast<int>(ranking.size()); ++i)
        std::cout << "  " << (i + 1) << ". " << ranking[i].second
                  << " (loss: " << std::fixed << std::setprecision(6)
                  << ranking[i].first << ")\n";

    std::cout << "\n=== Optimizer Comparison Complete ===\n";
    return 0;
}
