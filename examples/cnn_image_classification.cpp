/**
 * @file cnn_image_classification.cpp
 * @brief Convolutional Neural Network for image classification on synthetic data.
 *
 * Architecture:
 *   Conv2D(1, 4, 3, pad=1) -> ReLU -> MaxPool2D(2) -> Flatten -> Linear(64, 2)
 *
 * Loss:     SoftmaxCrossEntropy
 * Optimizer: Adam (built-in for all layers)
 * Data:     Synthetic 8x8 grayscale images (class 0: horizontal bars, class 1: vertical bars)
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

// Generate synthetic data centered around zero for numerical stability
static void generate_stripe_data(
    Eigen::Tensor<float, 4>& images,
    Eigen::Tensor<float, 2>& labels,
    int samples_per_class,
    float noise = 0.05f)
{
    const int N = 2 * samples_per_class;
    const int H = 8, W = 8;
    images.resize(N, 1, H, W);
    labels.resize(N, 2);
    images.setZero();
    labels.setZero();

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, noise);
    int idx = 0;

    // Class 0: horizontal bars
    for (int s = 0; s < samples_per_class; ++s)
    {
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                images(idx, 0, r, c) = ((r % 2 == 0) ? 0.5f : -0.5f) + nd(rng);
        labels(idx, 0) = 1.0f;
        ++idx;
    }

    // Class 1: vertical bars
    for (int s = 0; s < samples_per_class; ++s)
    {
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                images(idx, 0, r, c) = ((c % 2 == 0) ? 0.5f : -0.5f) + nd(rng);
        labels(idx, 1) = 1.0f;
        ++idx;
    }
}

static float compute_accuracy(const Eigen::Tensor<float, 2>& logits,
                              const Eigen::Tensor<float, 2>& targets)
{
    int batch = logits.dimension(0);
    int classes = logits.dimension(1);
    int correct = 0;
    for (int b = 0; b < batch; ++b)
    {
        int pred = 0, gt = 0;
        float pm = logits(b, 0), tm = targets(b, 0);
        for (int c = 1; c < classes; ++c)
        {
            if (logits(b, c) > pm) { pm = logits(b, c); pred = c; }
            if (targets(b, c) > tm) { tm = targets(b, c); gt = c; }
        }
        if (pred == gt) ++correct;
    }
    return static_cast<float>(correct) / batch * 100.0f;
}

int main()
{
    std::cout << "=== CNN Image Classification (Horizontal vs Vertical Stripes) ===\n\n";

    const int   samples_per_class = 200;
    const int   N                 = 2 * samples_per_class;
    const int   num_classes       = 2;
    const int   batch_size        = 32;
    const int   epochs            = 60;
    const float learning_rate     = 0.001f;

    Eigen::Tensor<float, 4> images;
    Eigen::Tensor<float, 2> labels;
    generate_stripe_data(images, labels, samples_per_class);
    std::cout << "Dataset: " << N << " images, 8x8 grayscale, " << num_classes << " classes\n";

    // Layers: Conv2D(1,4,3,pad=1) -> ReLU -> MaxPool(2) -> Flatten -> Linear(64,2)
    auto conv1   = std::make_shared<Layers::Conv2D>(1, 4, 3, 1, 1, true, "cpu-eigen");
    auto pool1   = std::make_shared<Layers::MaxPool2D>(2, -1, "cpu-eigen");
    auto flatten = std::make_shared<Layers::Flatten>();
    auto fc      = std::make_shared<Layers::Linear>(64, num_classes, "classifier",
                                                    true, true, "cpu-eigen");

    Activations::ReLU relu1("cpu-eigen");

    Models::SequentialModel model;
    model.add_layer(conv1);
    model.add_layer(pool1);
    model.add_layer(flatten);
    model.add_layer(fc);

    Losses::SoftmaxCrossEntropy loss;
    Optimizers::Adam optimizer;

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(99);

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
            Eigen::Tensor<float, 4> x_batch(batch_size, 1, 8, 8);
            Eigen::Tensor<float, 2> y_batch(batch_size, num_classes);
            for (int i = 0; i < batch_size; ++i)
            {
                int idx = indices[start + i];
                for (int r = 0; r < 8; ++r)
                    for (int c = 0; c < 8; ++c)
                        x_batch(i, 0, r, c) = images(idx, 0, r, c);
                for (int cl = 0; cl < num_classes; ++cl)
                    y_batch(i, cl) = labels(idx, cl);
            }

            // Forward
            auto z1  = conv1->forward(x_batch);
            auto a1  = relu1.forward(z1);
            auto p1  = pool1->forward(a1);
            auto f1  = flatten->forward(p1);
            auto out = fc->forward(f1);

            float batch_loss = loss.forward(out, y_batch);
            epoch_loss += batch_loss;
            epoch_acc  += compute_accuracy(out, y_batch);
            ++num_batches;

            // Backward
            auto grad = loss.backward(out, y_batch);
            grad = fc->backward(grad);
            auto grad4d = flatten->backward(grad);
            grad4d = pool1->backward(grad4d);
            grad4d = relu1.backward(grad4d);
            conv1->backward(grad4d);

            // Update
            model.update(optimizer, learning_rate);
            fc->reset_grads();
        }

        if ((epoch + 1) % 5 == 0 || epoch == 0)
        {
            std::cout << std::fixed << std::setprecision(4)
                      << "Epoch " << std::setw(3) << epoch + 1
                      << "  |  Loss: " << epoch_loss / num_batches
                      << "  |  Accuracy: " << epoch_acc / num_batches << "%\n";
        }
    }

    // Evaluate
    int total_correct = 0;
    for (int start = 0; start < N; start += batch_size)
    {
        int bs = std::min(batch_size, N - start);
        Eigen::Tensor<float, 4> x_eval(bs, 1, 8, 8);
        Eigen::Tensor<float, 2> y_eval(bs, num_classes);
        for (int i = 0; i < bs; ++i)
        {
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c)
                    x_eval(i, 0, r, c) = images(start + i, 0, r, c);
            for (int cl = 0; cl < num_classes; ++cl)
                y_eval(i, cl) = labels(start + i, cl);
        }

        auto z1  = conv1->forward(x_eval);
        auto a1  = relu1.forward(z1);
        auto p1  = pool1->forward(a1);
        auto f1  = flatten->forward(p1);
        auto out = fc->forward(f1);

        for (int b = 0; b < bs; ++b)
        {
            int pred = (out(b, 0) > out(b, 1)) ? 0 : 1;
            int gt = (y_eval(b, 0) > y_eval(b, 1)) ? 0 : 1;
            if (pred == gt) ++total_correct;
        }
    }

    std::cout << "\nFinal accuracy on full dataset: " << std::fixed << std::setprecision(2)
              << static_cast<float>(total_correct) / N * 100.0f << "%\n";
    std::cout << "\n=== CNN Example Complete ===\n";
    return 0;
}
