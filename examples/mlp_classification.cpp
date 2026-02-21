/**
 * @file mlp_classification.cpp
 * @brief Multi-Layer Perceptron for multi-class classification on synthetic spiral data.
 *
 * Architecture:
 *   Linear(2, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 3)
 *
 * Loss:     SoftmaxCrossEntropy (fused log-softmax + NLL)
 * Optimizer: Adam
 * Data:     3-class spiral dataset (synthetic, 2D features)
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

// ---------------------------------------------------------------------------
// Generate a 3-class spiral dataset in 2D
// ---------------------------------------------------------------------------
static void generate_spiral_data(
    Eigen::Tensor<float, 2>& X,       // [N, 2]
    Eigen::Tensor<float, 2>& Y_onehot,// [N, 3] one-hot
    int samples_per_class,
    float noise = 0.20f)
{
    const int num_classes = 3;
    const int N = samples_per_class * num_classes;

    X.resize(N, 2);
    Y_onehot.resize(N, num_classes);
    Y_onehot.setZero();

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, noise);

    int idx = 0;
    for (int c = 0; c < num_classes; ++c)
    {
        for (int i = 0; i < samples_per_class; ++i)
        {
            float r = static_cast<float>(i) / samples_per_class;
            float theta = 4.0f * r + static_cast<float>(c) * 2.0f * M_PI / num_classes;
            X(idx, 0) = r * std::cos(theta) + nd(rng);
            X(idx, 1) = r * std::sin(theta) + nd(rng);
            Y_onehot(idx, c) = 1.0f;
            ++idx;
        }
    }
}

// ---------------------------------------------------------------------------
// Compute accuracy from logits vs one-hot targets
// ---------------------------------------------------------------------------
static float compute_accuracy(const Eigen::Tensor<float, 2>& logits,
                              const Eigen::Tensor<float, 2>& targets)
{
    int batch = logits.dimension(0);
    int classes = logits.dimension(1);
    int correct = 0;
    for (int b = 0; b < batch; ++b)
    {
        int pred_class = 0, true_class = 0;
        float pred_max = logits(b, 0), true_max = targets(b, 0);
        for (int c = 1; c < classes; ++c)
        {
            if (logits(b, c) > pred_max) { pred_max = logits(b, c); pred_class = c; }
            if (targets(b, c) > true_max) { true_max = targets(b, c); true_class = c; }
        }
        if (pred_class == true_class) ++correct;
    }
    return static_cast<float>(correct) / batch * 100.0f;
}

// ===========================================================================
int main()
{
    std::cout << "=== MLP Multi-Class Classification (Spiral Dataset) ===\n\n";

    // ---- Hyperparameters ----
    const int   samples_per_class = 200;
    const int   num_classes       = 3;
    const int   N                 = samples_per_class * num_classes;  // 600
    const int   batch_size        = 32;
    const int   epochs            = 80;
    const float learning_rate     = 0.01f;

    // ---- Data ----
    Eigen::Tensor<float, 2> X, Y;
    generate_spiral_data(X, Y, samples_per_class);
    std::cout << "Dataset: " << N << " samples, " << num_classes << " classes\n";

    // ---- Layers ----
    auto linear1 = std::make_shared<Layers::Linear>(2, 64, "fc1", true, true, "cpu-eigen");
    auto linear2 = std::make_shared<Layers::Linear>(64, 32, "fc2", true, true, "cpu-eigen");
    auto linear3 = std::make_shared<Layers::Linear>(32, num_classes, "fc3", true, true, "cpu-eigen");

    Activations::ReLU relu1("cpu-eigen");
    Activations::ReLU relu2("cpu-eigen");

    // ---- Model (for update()) ----
    Models::SequentialModel model;
    model.add_layer(linear1);
    model.add_layer(linear2);
    model.add_layer(linear3);

    // ---- Loss & Optimizer ----
    Losses::SoftmaxCrossEntropy loss;
    Optimizers::Adam optimizer;

    // ---- Shuffled index array ----
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123);

    // ---- Training loop ----
    model.summary();
    std::cout << "\n";

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        float epoch_acc  = 0.0f;
        int   num_batches = 0;

        for (int start = 0; start + batch_size <= N; start += batch_size)
        {
            int bs = batch_size;

            // ---- Gather mini-batch ----
            Eigen::Tensor<float, 2> x_batch(bs, 2);
            Eigen::Tensor<float, 2> y_batch(bs, num_classes);
            for (int i = 0; i < bs; ++i)
            {
                int idx = indices[start + i];
                for (int f = 0; f < 2; ++f)
                    x_batch(i, f) = X(idx, f);
                for (int c = 0; c < num_classes; ++c)
                    y_batch(i, c) = Y(idx, c);
            }

            // ---- Forward pass ----
            auto z1 = linear1->forward(x_batch);  // [bs, 64]
            auto a1 = relu1.forward(z1);           // [bs, 64]
            auto z2 = linear2->forward(a1);        // [bs, 32]
            auto a2 = relu2.forward(z2);           // [bs, 32]
            auto z3 = linear3->forward(a2);        // [bs, 3] (logits)

            // ---- Loss ----
            float batch_loss = loss.forward(z3, y_batch);
            epoch_loss += batch_loss;
            epoch_acc  += compute_accuracy(z3, y_batch);
            ++num_batches;

            // ---- Backward pass ----
            auto grad = loss.backward(z3, y_batch);  // [bs, 3]
            grad = linear3->backward(grad);           // [bs, 32]
            grad = relu2.backward(grad);              // [bs, 32]
            grad = linear2->backward(grad);           // [bs, 64]
            grad = relu1.backward(grad);              // [bs, 64]
            linear1->backward(grad);                  // [bs, 2]

            // ---- Weight update ----
            model.update(optimizer, learning_rate);

            // ---- Reset gradients (they accumulate by default) ----
            linear1->reset_grads();
            linear2->reset_grads();
            linear3->reset_grads();
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0)
        {
            std::cout << std::fixed << std::setprecision(4)
                      << "Epoch " << std::setw(3) << epoch + 1
                      << "  |  Loss: " << epoch_loss / num_batches
                      << "  |  Accuracy: " << epoch_acc / num_batches << "%\n";
        }
    }

    // ---- Evaluation on full dataset ----
    auto z1 = linear1->forward(X);
    auto a1 = relu1.forward(z1);
    auto z2 = linear2->forward(a1);
    auto a2 = relu2.forward(z2);
    auto z3 = linear3->forward(a2);

    float final_acc = compute_accuracy(z3, Y);
    std::cout << "\nFinal accuracy on full dataset: " << std::fixed << std::setprecision(2)
              << final_acc << "%\n";

    std::cout << "\n=== MLP Example Complete ===\n";
    return 0;
}
