/**
 * @file regularized_cnn.cpp
 * @brief CNN with BatchNorm, Dropout, and LeakyReLU on synthetic image data.
 *
 * Architecture:
 *   Conv2D(1,8,3,pad=1) -> LeakyReLU -> BatchNorm(128) -> MaxPool2D(2) ->
 *   Flatten -> Dropout(0.3) -> Linear(128,3)
 *
 * Loss:     CategoricalCrossEntropy (with integer class targets)
 * Optimizer: Adagrad
 * Data:     Synthetic 8x8 grayscale images (3 classes: horizontal, vertical, diagonal)
 *
 * Demonstrates: Conv2D, BatchNorm, Dropout, LeakyReLU, MaxPool2D,
 *               CategoricalCrossEntropy, Adagrad
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

static void generate_pattern_data(
    Eigen::Tensor<float, 4>& images,
    Eigen::Tensor<int, 1>&   labels,
    int samples_per_class,
    float noise = 0.05f)
{
    const int num_classes = 3;
    const int N = num_classes * samples_per_class;
    const int H = 8, W = 8;
    images.resize(N, 1, H, W);
    labels.resize(N);
    images.setZero();

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, noise);
    int idx = 0;

    // Class 0: horizontal bars
    for (int s = 0; s < samples_per_class; ++s)
    {
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                images(idx, 0, r, c) = ((r % 2 == 0) ? 0.5f : -0.5f) + nd(rng);
        labels(idx) = 0;
        ++idx;
    }

    // Class 1: vertical bars
    for (int s = 0; s < samples_per_class; ++s)
    {
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                images(idx, 0, r, c) = ((c % 2 == 0) ? 0.5f : -0.5f) + nd(rng);
        labels(idx) = 1;
        ++idx;
    }

    // Class 2: diagonal pattern
    for (int s = 0; s < samples_per_class; ++s)
    {
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                images(idx, 0, r, c) = (((r + c) % 2 == 0) ? 0.5f : -0.5f) + nd(rng);
        labels(idx) = 2;
        ++idx;
    }
}

static float compute_accuracy(const Eigen::Tensor<float, 2>& logits,
                              const Eigen::Tensor<int, 1>&   targets)
{
    int batch = logits.dimension(0);
    int classes = logits.dimension(1);
    int correct = 0;
    for (int b = 0; b < batch; ++b)
    {
        int pred_class = 0;
        float pred_max = logits(b, 0);
        for (int c = 1; c < classes; ++c)
            if (logits(b, c) > pred_max) { pred_max = logits(b, c); pred_class = c; }
        if (pred_class == targets(b)) ++correct;
    }
    return static_cast<float>(correct) / batch * 100.0f;
}

int main()
{
    std::cout << "=== Regularized CNN (BatchNorm + Dropout + LeakyReLU) ===\n\n";

    const int   samples_per_class = 200;
    const int   num_classes       = 3;
    const int   N                 = num_classes * samples_per_class;
    const int   batch_size        = 32;
    const int   epochs            = 60;
    const float learning_rate     = 0.01f;

    Eigen::Tensor<float, 4> images;
    Eigen::Tensor<int, 1>   labels;
    generate_pattern_data(images, labels, samples_per_class);
    std::cout << "Dataset: " << N << " images (8x8), " << num_classes << " classes\n";

    // Conv2D(1,8,3,stride=1,pad=1) -> 8 channels, 8x8 output
    // MaxPool2D(2) -> 8 channels, 4x4 = 128 features after flatten
    auto conv1    = std::make_shared<Layers::Conv2D>(1, 8, 3, 1, 1, true, "cpu-eigen");
    auto pool1    = std::make_shared<Layers::MaxPool2D>(2, -1, "cpu-eigen");
    auto flatten  = std::make_shared<Layers::Flatten>();
    auto fc       = std::make_shared<Layers::Linear>(128, num_classes, "classifier",
                                                      true, true, "cpu-eigen");

    Activations::LeakyReLU leaky_relu(0.01f, "cpu-eigen");
    Layers::BatchNorm      batch_norm(128);
    Layers::Dropout        dropout(0.3f);

    Models::SequentialModel model;
    model.add_layer(conv1);
    model.add_layer(pool1);
    model.add_layer(flatten);
    model.add_layer(fc);

    Losses::CategoricalCrossEntropy loss("mean", true);
    Optimizers::Adagrad optimizer;

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(99);

    model.summary();
    std::cout << "\n";

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        batch_norm.train();
        dropout.train();

        float epoch_loss = 0.0f;
        float epoch_acc  = 0.0f;
        int   num_batches = 0;

        for (int start = 0; start + batch_size <= N; start += batch_size)
        {
            Eigen::Tensor<float, 4> x_batch(batch_size, 1, 8, 8);
            Eigen::Tensor<int, 1>   y_batch(batch_size);
            for (int i = 0; i < batch_size; ++i)
            {
                int idx = indices[start + i];
                for (int r = 0; r < 8; ++r)
                    for (int c = 0; c < 8; ++c)
                        x_batch(i, 0, r, c) = images(idx, 0, r, c);
                y_batch(i) = labels(idx);
            }

            // Forward: Conv -> LeakyReLU -> Pool -> Flatten -> BatchNorm -> Dropout -> FC
            auto z1 = conv1->forward(x_batch);
            auto a1 = leaky_relu.forward(z1);
            auto p1 = pool1->forward(a1);
            auto f1 = flatten->forward(p1);
            auto bn = batch_norm.forward(f1);
            auto dr = dropout.forward(bn);
            auto out = fc->forward(dr);

            float batch_loss = loss.forward(out, y_batch);
            epoch_loss += batch_loss;
            epoch_acc  += compute_accuracy(out, y_batch);
            ++num_batches;

            // Backward
            auto grad = loss.backward(out, y_batch);
            grad = fc->backward(grad);
            grad = dropout.backward(grad);
            grad = batch_norm.backward(grad);
            auto grad4d = flatten->backward(grad);
            grad4d = pool1->backward(grad4d);
            grad4d = leaky_relu.backward(grad4d);
            conv1->backward(grad4d);

            // Update
            model.update(optimizer, learning_rate);
            batch_norm.step(optimizer, learning_rate);
            fc->reset_grads();
            batch_norm.reset_grads();
        }

        if ((epoch + 1) % 5 == 0 || epoch == 0)
        {
            std::cout << std::fixed << std::setprecision(4)
                      << "Epoch " << std::setw(3) << epoch + 1
                      << "  |  Loss: " << epoch_loss / num_batches
                      << "  |  Accuracy: " << epoch_acc / num_batches << "%\n";
        }
    }

    // Evaluate (no dropout in eval mode)
    batch_norm.eval();
    dropout.eval();

    int total_correct = 0;
    for (int start = 0; start < N; start += batch_size)
    {
        int bs = std::min(batch_size, N - start);
        Eigen::Tensor<float, 4> x_eval(bs, 1, 8, 8);
        Eigen::Tensor<int, 1>   y_eval(bs);
        for (int i = 0; i < bs; ++i)
        {
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c)
                    x_eval(i, 0, r, c) = images(start + i, 0, r, c);
            y_eval(i) = labels(start + i);
        }

        auto z1 = conv1->forward(x_eval);
        auto a1 = leaky_relu.forward(z1);
        auto p1 = pool1->forward(a1);
        auto f1 = flatten->forward(p1);
        auto bn = batch_norm.forward(f1);
        auto dr = dropout.forward(bn);
        auto out = fc->forward(dr);

        int classes = out.dimension(1);
        for (int b = 0; b < bs; ++b)
        {
            int pred = 0;
            float pm = out(b, 0);
            for (int c = 1; c < classes; ++c)
                if (out(b, c) > pm) { pm = out(b, c); pred = c; }
            if (pred == y_eval(b)) ++total_correct;
        }
    }

    std::cout << "\nFinal accuracy on full dataset: " << std::fixed << std::setprecision(2)
              << static_cast<float>(total_correct) / N * 100.0f << "%\n";
    std::cout << "\n=== Regularized CNN Example Complete ===\n";
    return 0;
}
