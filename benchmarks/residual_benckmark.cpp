/**
 * @file residual_benckmark.cpp
 * @brief Deep ResNet device benchmark — Spiral Classification
 *
 * Benchmarks the Residual (skip-connection) layer across all three
 * CppNet compute devices: cpu-eigen, cpu (OpenMP), and gpu (CUDA).
 *
 * Architecture (per config):
 *   Linear(2, W) → ReLU
 *   → [Residual(Linear(W,W) → Linear(W,W))] → ReLU   ×D blocks
 *   → Linear(W, 5)                                    (classifier)
 *
 * The residual blocks use identity shortcuts (in_features == out_features).
 * The depth parameter D controls the number of stacked residual blocks,
 * making the network progressively deeper while keeping width constant.
 *
 * Dataset: 2D spiral with 5 classes (same as MLP benchmark).
 */

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


// data generation 
static void generate_spiral_data(
    Eigen::Tensor<float, 2>& X,
    Eigen::Tensor<float, 2>& Y_onehot,
    int samples_per_class,
    int num_classes = 5,
    float noise = 0.05f)
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


// network configuration 
struct NetConfig
{
    std::string label;
    int width;              // hidden-layer width W
    int depth;              // number of residual blocks D
    int epochs;
    int batch_size;
    float lr;
};


// architecture string for display
static std::string arch_str(int width, int depth, int num_classes)
{
    std::string s = "2→" + std::to_string(width) + "→[ResBlock×"
                  + std::to_string(depth) + "]→" + std::to_string(num_classes);
    return s;
}


// training function for a single config-device pair; returns (time, final_loss, final_acc)
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

    if (dev == "cpu" || dev == "cpu-eigen")
    {
        CppNet::Layers::Linear::set_num_threads(4);
        CppNet::Activations::ReLU::set_num_threads(4);
    }

    // build the network
    // we use SequentialModel for the pipeline: proj → relu → [res+relu]×D → head
    CppNet::Models::SequentialModel model;

    // input projection: Linear(2, W)
    auto proj = std::make_shared<CppNet::Layers::Linear>(
        input_dim, cfg.width, "proj", true, true, dev, "he");
    #ifdef USE_CUDA
    if (dev == "gpu") proj->set_max_batch_size(cfg.batch_size);
    #endif
    model.add_layer(proj);
    model.add_activation(std::make_shared<CppNet::Activations::ReLU>(dev));

    // stacked residual blocks
    for (int d = 0; d < cfg.depth; ++d)
    {
        std::string tag = "res" + std::to_string(d);

        // build the inner block: Linear(W,W) → Linear(W,W)
        auto fc_a = std::make_shared<CppNet::Layers::Linear>(
            cfg.width, cfg.width, tag + "_a", true, true, dev, "he");
        auto fc_b = std::make_shared<CppNet::Layers::Linear>(
            cfg.width, cfg.width, tag + "_b", true, true, dev, "he");

        #ifdef USE_CUDA
        if (dev == "gpu")
        {
            fc_a->set_max_batch_size(cfg.batch_size);
            fc_b->set_max_batch_size(cfg.batch_size);
        }
        #endif

        std::vector<std::shared_ptr<CppNet::Layers::Layer>> block = {fc_a, fc_b};

        auto res = std::make_shared<CppNet::Layers::Residual>(
            block, cfg.width, cfg.width, dev);

        model.add_layer(res);
        model.add_activation(std::make_shared<CppNet::Activations::ReLU>(dev));
    }

    // output head: Linear(W, num_classes)
    auto head = std::make_shared<CppNet::Layers::Linear>(
        cfg.width, num_classes, "head", true, true, dev, "xavier");
    #ifdef USE_CUDA
    if (dev == "gpu") head->set_max_batch_size(cfg.batch_size);
    #endif
    model.add_layer(head);

    CppNet::Losses::SoftmaxCrossEntropy loss("mean");
    CppNet::Optimizers::Adam optim(0.9f, 0.999f, 1e-10f);

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123);

    float final_loss = 0.0f, final_acc = 0.0f;

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

        if (verbose && ((epoch + 1) % std::max(1, cfg.epochs / 5) == 0 || epoch == 0))
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


int main()
{
    std::cout << "------------------------------------------------------------------\n";
    std::cout << "-   CppNet — Deep ResNet Device Benchmark (Spiral Classification)-\n";
    std::cout << "------------------------------------------------------------------\n\n";

    const int samples_per_class = 3000;
    const int num_classes       = 5;
    const int N = samples_per_class * num_classes;

    Eigen::Tensor<float, 2> X, Y;
    generate_spiral_data(X, Y, samples_per_class, num_classes, 0.05f);
    std::cout << "Dataset: " << N << " samples, " << num_classes
              << " classes, 2 features\n\n";

    std::vector<NetConfig> configs = {
        {"Small",    64,   2,   20,  128,  0.001f},    // 2→64→[ResBlock×2]→5
        {"Medium",  128,   4,   10,  256,  0.0005f},   // 2→128→[ResBlock×4]→5
        {"Large",   256,   6,   10,  256,  0.0005f},   // 2→256→[ResBlock×6]→5
    };

    std::vector<std::string> devices = {"cpu-eigen", "cpu"};
    #ifdef USE_CUDA
    devices.push_back("gpu");
    #endif

    struct Result { double ms; float loss; float acc; };
    std::map<std::string, std::map<std::string, Result>> results;

    for (auto& cfg : configs)
    {
        std::string arch = arch_str(cfg.width, cfg.depth, num_classes);
        std::cout << "---------------------------------------------------------------\n";
        std::cout << "Config: " << cfg.label
                  << "   |   " << arch
                  << "\n  width=" << cfg.width
                  << ", depth=" << cfg.depth
                  << ", epochs=" << cfg.epochs
                  << ", batch=" << cfg.batch_size
                  << ", N=" << N << "\n";
        std::cout << "---------------------------------------------------------------\n";

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

    // summary table
    std::cout << "\n\n";
    std::cout << "---------------------------------------------------------------\n";
    std::cout << "-              Deep ResNet Benchmark — Summary                   -\n";
    std::cout << "---------------------------------------------------------------\n\n";

    std::cout << std::left
              << std::setw(10) << "Config"
              << std::setw(35) << "Architecture";
    for (auto& dev : devices)
        std::cout << std::setw(15) << dev;
    std::cout << "\n";
    std::cout << std::string(10 + 35 + 15 * (int)devices.size(), '-') << "\n";

    for (auto& cfg : configs)
    {
        std::string arch = arch_str(cfg.width, cfg.depth, num_classes);
        std::cout << std::left
                  << std::setw(10) << cfg.label
                  << std::setw(35) << arch;
        for (auto& dev : devices)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2)
                << results[cfg.label][dev].ms / 1000.0 << " s";
            std::cout << std::setw(15) << oss.str();
        }
        std::cout << "\n";
    }

    // speedup vs cpu-eigen 
    std::cout << "\n-  Speedup vs cpu-eigen  -\n";
    for (auto& cfg : configs)
    {
        double baseline = results[cfg.label]["cpu-eigen"].ms;
        std::cout << "  " << std::left << std::setw(10) << cfg.label << ":  ";
        for (auto& dev : devices)
        {
            double speedup = baseline / results[cfg.label][dev].ms;
            std::cout << dev << "=" << std::fixed << std::setprecision(2)
                      << speedup << "x  ";
        }
        std::cout << "\n";
    }

    // final accuracy 
    std::cout << "\n── Final Accuracy ──\n";
    for (auto& cfg : configs)
    {
        std::cout << "  " << std::left << std::setw(10) << cfg.label << ":  ";
        for (auto& dev : devices)
            std::cout << dev << "=" << std::fixed << std::setprecision(1)
                      << results[cfg.label][dev].acc << "%  ";
        std::cout << "\n";
    }

    // final loss
    std::cout << "\n── Final Loss ──\n";
    for (auto& cfg : configs)
    {
        std::cout << "  " << std::left << std::setw(10) << cfg.label << ":  ";
        for (auto& dev : devices)
            std::cout << dev << "=" << std::fixed << std::setprecision(4)
                      << results[cfg.label][dev].loss << "  ";
        std::cout << "\n";
    }

    std::cout << "\n-  Deep ResNet Benchmark Complete  -\n";
    return 0;
}
