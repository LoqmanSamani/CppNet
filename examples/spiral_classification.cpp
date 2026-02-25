#include <CppNet/CppNet.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <tuple>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;



// data generation: create a 2D spiral dataset with given samples/class, classes, and noise level
static void generate_spiral_data(
    Eigen::Tensor<float, 2>& X,
    Eigen::Tensor<float, 2>& Y_onehot,
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

void plot_spiral(const Eigen::Tensor<float, 2>& X,
                 const Eigen::Tensor<float, 2>& Y_onehot)
{
    const int N = X.dimension(0);
    const int num_classes = Y_onehot.dimension(1);

    std::vector<std::vector<double>> x_vals(num_classes);
    std::vector<std::vector<double>> y_vals(num_classes);

    for (int i = 0; i < N; ++i)
    {
        int label = 0;
        for (int c = 0; c < num_classes; ++c)
            if (Y_onehot(i, c) == 1.0f) { label = c; break; }
        x_vals[label].push_back(X(i, 0));
        y_vals[label].push_back(X(i, 1));
    }

    for (int c = 0; c < num_classes; ++c)
        plt::scatter(x_vals[c], y_vals[c], 10.0);
    plt::title("Spiral Dataset");
    plt::show();
}


// network configuration struct: holds hyperparameters for a single experiment
struct NetConfig
{
    std::string label;           // human-readable tag
    std::vector<int> widths;     // hidden-layer widths (input & output added automatically)
    int epochs;
    int batch_size;
    float lr;
};

// training configuration helper: builds the model, runs training, and returns timing, final loss, and accuracy
static std::tuple<double, float, float> train_config(
    const NetConfig& cfg,
    const std::string& dev,
    const Eigen::Tensor<float, 2>& X,
    const Eigen::Tensor<float, 2>& Y,
    int num_classes,
    bool verbose)
{
    const int N = X.dimension(0);
    const int input_dim = X.dimension(1);

    // build full width vector: input_dim -> hidden... -> num_classes
    std::vector<int> dims;
    dims.push_back(input_dim);
    for (int w : cfg.widths) dims.push_back(w);
    dims.push_back(num_classes);

    // set OpenMP threads for CPU backends
    if (dev == "cpu" || dev == "cpu-eigen")
    {
        CppNet::Layers::Linear::set_num_threads(4);
        CppNet::Activations::ReLU::set_num_threads(4);
    }

    // build the network
    CppNet::Models::SequentialModel model;
    std::vector<std::shared_ptr<CppNet::Layers::Linear>> layers;

    for (int i = 0; i + 1 < (int)dims.size(); ++i)
    {
        std::string name = "fc" + std::to_string(i + 1);
        auto fc = std::make_shared<CppNet::Layers::Linear>(
            dims[i], dims[i + 1], name, true, true, dev, "xavier");

        #ifdef USE_CUDA
        if (dev == "gpu")
            fc->set_max_batch_size(cfg.batch_size);
        #endif

        model.add_layer(fc);
        layers.push_back(fc);

        // relu activation after every layer except the last
        if (i + 2 < (int)dims.size())
            model.add_activation(std::make_shared<CppNet::Activations::ReLU>(dev));
    }

    CppNet::Losses::SoftmaxCrossEntropy loss("mean");
    CppNet::Optimizers::Adam optim(0.9f, 0.999f, 1e-10f);

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123);

    float final_loss = 0.0f, final_acc = 0.0f;

    // elapsed time will include all epochs, shuffling, batching, forward/backward passes, and updates
    CppNet::Utils::Timer timer(dev, false);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f, epoch_acc = 0.0f;
        int num_batches = 0;

        for (int start = 0; start + cfg.batch_size <= N; start += cfg.batch_size)
        {
            int bs = cfg.batch_size;

            Eigen::Tensor<float, 2> x_batch(bs, input_dim);
            Eigen::Tensor<float, 2> y_batch(bs, num_classes);
            for (int i = 0; i < bs; ++i)
            {
                int idx = indices[start + i];
                for (int f = 0; f < input_dim; ++f)
                    x_batch(i, f) = X(idx, f);
                for (int c = 0; c < num_classes; ++c)
                    y_batch(i, c) = Y(idx, c);
            }

            model.zero_grad();
            auto output = model.forward(x_batch);
            auto batch_loss = loss.forward(output, y_batch);
            epoch_loss += batch_loss;
            epoch_acc  += CppNet::Metrics::accuracy(output, y_batch);
            ++num_batches;

            auto grad_loss = loss.backward(output, y_batch);
            model.backward(grad_loss);
            model.update(optim, cfg.lr);
        }

        final_loss = epoch_loss / num_batches;
        final_acc  = epoch_acc  / num_batches;

        if (verbose && ((epoch + 1) % 20 == 0 || epoch == 0))
        {
            std::cout << std::fixed << std::setprecision(4)
                      << "    Epoch " << std::setw(3) << epoch + 1
                      << "  |  Loss: " << final_loss
                      << "  |  Accuracy: " << final_acc << "%\n";
        }
    }

    double elapsed_ms = timer.elapsed_ms();
    return {elapsed_ms, final_loss, final_acc};
}


// helper to format architecture string for display
static std::string arch_str(int in, const std::vector<int>& widths, int out)
{
    std::string s = std::to_string(in);
    for (int w : widths) s += "→" + std::to_string(w);
    s += "→" + std::to_string(out);
    return s;
}



int main()
{
    std::cout << "---------------------------------------------------------------\n";
    std::cout << "    CppNet — MLP Device Benchmark (Spiral Classification)      \n";
    std::cout << "---------------------------------------------------------------\n";

    const int samples_per_class = 3000;
    const int num_classes       = 5;
    const int N = samples_per_class * num_classes;

    Eigen::Tensor<float, 2> X, Y;
    generate_spiral_data(X, Y, samples_per_class, num_classes, 0.05f);
    std::cout << "Dataset: " << N << " samples, " << num_classes << " classes, 2 features\n";

    plot_spiral(X, Y);

    // network configurations (small → xlarge)
    //
    //  The goal is to sweep tensor sizes so the GPU matmul kernels
    //  operate on progressively larger matrices.  for a batch of B
    //  and hidden width H the dominant matmul is  B×H * H×H.
    //
    //  small layers ≈ Eigen is fast, GPU overhead dominates.
    //  large layers ≈ GPU throughput wins.

    std::vector<NetConfig> configs = {
        {"Small",    {64,  64},                  50,  128, 0.001f},
        {"Medium",   {128, 256, 128},            50,  256, 0.0005f},
        {"Large",    {256, 512, 512, 256},       30,  256, 0.0005f},
        {"XLarge",   {512, 1024, 1024, 512},     20,  512, 0.0003f},
    };

    // devices to benchmark
    std::vector<std::string> devices = {"cpu-eigen", "cpu"};
    #ifdef USE_CUDA
    devices.push_back("gpu");
    #endif

    // results storage: [config_label][device] = (ms, loss, acc)
    struct Result { double ms; float loss; float acc; };
    std::map<std::string, std::map<std::string, Result>> results;

    // run all experiments
    for (auto& cfg : configs)
    {
        std::string arch = arch_str(2, cfg.widths, num_classes);
        std::cout << "\n----------------------------------------------------\n";
        std::cout << "  Config: " << cfg.label
                  << "   |   Architecture: " << arch
                  << "   |   Epochs: " << cfg.epochs
                  << "   |   Batch: " << cfg.batch_size << "\n";
        std::cout << "----------------------------------------------------\n";

        for (auto& dev : devices)
        {
            std::cout << "\n  ▸ Device: " << dev << "\n";
            auto [ms, loss, acc] = train_config(cfg, dev, X, Y, num_classes, true);
            results[cfg.label][dev] = {ms, loss, acc};

            std::cout << std::fixed << std::setprecision(2)
                      << "    ✓ Time: " << ms << " ms ("
                      << ms / 1000.0 << " s)   Loss: "
                      << std::setprecision(4) << loss
                      << "   Acc: " << acc << "%\n";
        }
    }

    
    // summary table: timing results and speedups relative to cpu-eigen baseline
    std::cout << "\n\n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";
    std::cout << "                          CppNet Device Benchmark — Summary                                       \n";
    std::cout << "--------------------------------------------------------------------------------------------------\n";
    std::cout << "-  Config   -       Architecture            -";

    for (auto& dev : devices)
        std::cout << "  " << std::left << std::setw(14) << dev << "║";
    std::cout << "\n";

    std::cout << "---------------------------------------------\n";
    for (size_t i = 0; i < devices.size(); ++i)
        std::cout << "-------------------\n";
    std::cout << "\n";

    for (auto& cfg : configs)
    {
        std::string arch = arch_str(2, cfg.widths, num_classes);
        std::cout << "║ " << std::left << std::setw(10) << cfg.label
                  << "║ " << std::setw(29) << arch << "║";

        for (auto& dev : devices)
        {
            auto& r = results[cfg.label][dev];
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << r.ms / 1000.0 << " s";
            std::cout << "  " << std::right << std::setw(12) << oss.str() << "  ║";
        }
        std::cout << "\n";
    }

    std::cout << "---------------------------------------------\n";
    for (size_t i = 0; i < devices.size(); ++i)
        std::cout << "-------------------\n";
    std::cout << "\n";

    // speedup rows (relative to cpu-eigen)
    std::cout << "-           -  Speedup vs cpu-eigen        -";
    for (auto& dev : devices)
    {
        if (dev == "cpu-eigen")
            std::cout << "    baseline    -";
        else
            std::cout << "                -";
    }
    std::cout << "\n";

    for (auto& cfg : configs)
    {
        double baseline = results[cfg.label]["cpu-eigen"].ms;
        std::cout << "- " << std::left << std::setw(10) << cfg.label
                  << "-                              -";

        for (auto& dev : devices)
        {
            if (dev == "cpu-eigen")
            {
                std::cout << "      1.00x     -";
            }
            else
            {
                double speedup = baseline / results[cfg.label][dev].ms;
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << speedup << "x";
                std::cout << "      " << std::left << std::setw(10) << oss.str() << "-";
            }
        }
        std::cout << "\n";
    }

    std::cout << "---------------------------------------------\n";
    for (size_t i = 0; i < devices.size(); ++i)
        std::cout << "-------------------\n";
    std::cout << "\n";

    // final accuracy summary (should be similar across devices since same training config and random seed)
    std::cout << "\n";
    std::cout << "Final accuracy per config (all devices should converge similarly):\n";
    for (auto& cfg : configs)
    {
        std::cout << "  " << std::left << std::setw(10) << cfg.label << ":  ";
        for (auto& dev : devices)
        {
            std::cout << dev << "=" << std::fixed << std::setprecision(1)
                      << results[cfg.label][dev].acc << "%  ";
        }
        std::cout << "\n";
    }

    return 0;
}