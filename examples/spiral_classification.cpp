#include <CppNet/CppNet.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;




static void generate_spiral_data(
    Eigen::Tensor<float, 2>& X,       // [N, 2]
    Eigen::Tensor<float, 2>& Y_onehot,// [N, num_classes] one-hot
    int samples_per_class,
    int num_classes = 3,
    float noise = 0.20f)
{
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



void plot_spiral(const Eigen::Tensor<float,2>& X,
                 const Eigen::Tensor<float,2>& Y_onehot)
{
    const int N = X.dimension(0);
    const int num_classes = Y_onehot.dimension(1);

    std::vector<std::vector<double>> x_vals(num_classes);
    std::vector<std::vector<double>> y_vals(num_classes);

    for (int i = 0; i < N; ++i)
    {
        int label = 0;
        for (int c = 0; c < num_classes; ++c)
        {
            if (Y_onehot(i, c) == 1.0f)
            {
                label = c;
                break;
            }
        }

        x_vals[label].push_back(X(i,0));
        y_vals[label].push_back(X(i,1));
    }

    for (int c = 0; c < num_classes; ++c)
        plt::scatter(x_vals[c], y_vals[c], 10.0);

    plt::title("Spiral Dataset");
    plt::show();
}


int main()
{
    std::cout << "=== MLP Multi-Class Classification (Spiral Dataset) ===\n\n";

    const int samples_per_class = 1000;
    const int num_classes = 5;
    const int N = samples_per_class * num_classes;
    const int epochs = 1000;
    const int batch_size = 128;
    const float learning_rate = 0.001f;
    const int log_freq = 50;

    Eigen::Tensor<float, 2> X, Y;
    generate_spiral_data(X, Y, samples_per_class, num_classes, 0.05f);
    std::cout << "Dataset: " << N << " samples, " << num_classes << " classes\n";

    plot_spiral(X, Y); // visualize the dataset

    // build a network (feed-forward network to classify the spiral data)
    CppNet::Models::SequentialModel model;
    model.add_layer(std::make_shared<CppNet::Layers::Linear>(2, 64, "fc1", true, true, "cpu-eigen", "xavier"));
    model.add_activation(std::make_shared<CppNet::Activations::ReLU>("cpu-eigen"));
    model.add_layer(std::make_shared<CppNet::Layers::Linear>(64, 128, "fc2", true, true, "cpu-eigen", "xavier"));
    model.add_activation(std::make_shared<CppNet::Activations::ReLU>("cpu-eigen"));
    //model.add_layer(std::make_shared<CppNet::Layers::Linear>(128, 256, "fc3", true, true, "cpu-eigen", "xavier"));
    //model.add_activation(std::make_shared<CppNet::Activations::ReLU>("cpu-eigen"));
    //model.add_layer(std::make_shared<CppNet::Layers::Linear>(256, 128, "fc4", true, true, "cpu-eigen", "xavier"));
    //model.add_activation(std::make_shared<CppNet::Activations::ReLU>("cpu-eigen"));
    model.add_layer(std::make_shared<CppNet::Layers::Linear>(128, 64, "fc5", true, true, "cpu-eigen", "xavier"));
    model.add_activation(std::make_shared<CppNet::Activations::ReLU>("cpu-eigen"));
    model.add_layer(std::make_shared<CppNet::Layers::Linear>(64, num_classes, "fc6", true, true, "cpu-eigen", "xavier"));
    model.add_activation(std::make_shared<CppNet::Activations::Softmax>("cpu-eigen"));

    // loss and optimizer
    CppNet::Losses::SoftmaxCrossEntropy loss("mean");
    CppNet::Optimizers::Adam optim(0.9f, 0.999f, 1e-10f);

    // shuffle indices for mini-batch sampling
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123);

    // model summary
    model.summary();
    std::cout << "\n";

    // training loop
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        float epoch_acc  = 0.0f;
        int   num_batches = 0;

        for (int start = 0; start + batch_size <= N; start += batch_size)
        {
            int bs = batch_size;

            // gather mini-batch
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

            // forward pass
            model.zero_grad(); // zero gradients before forward/backward
            auto output = model.forward(x_batch); // [bs, num_classes]
            auto batch_loss = loss.forward(output, y_batch);
            epoch_loss += batch_loss;
            epoch_acc  += CppNet::Metrics::accuracy(output, y_batch);
            ++num_batches;

            // backward pass and update
            auto grad_loss = loss.backward(output, y_batch); // [bs, num_classes]
            model.backward(grad_loss);
            model.update(optim, learning_rate);
        }

        // print epoch metrics every log_freq epochs
        if ((epoch + 1) % log_freq == 0 || epoch == 0)
        {
            std::cout << std::fixed << std::setprecision(4)
                      << "Epoch " << std::setw(3) << epoch + 1
                      << "  |  Loss: " << epoch_loss / num_batches
                      << "  |  Accuracy: " << epoch_acc / num_batches << "%\n";
        }
    }

    return 0;
}