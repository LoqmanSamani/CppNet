/**
 * @file cnn_benckmark.cpp
 * @brief CNN Device Benchmark — Image Classification
 *
 * Benchmarks Conv2D + MaxPool2D + Linear across cpu-eigen, cpu (OpenMP),
 * and gpu (CUDA) backends using synthetic CIFAR-10-shaped images
 * (3×32×32, 10 classes).
 *
 * The data is synthetic (random patterns per class) so the focus is on
 * throughput, not convergence quality.
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
#include <sstream>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


// ─── synthetic image dataset ─────────────────────────────────────────
// Each class gets a different deterministic pattern + noise so every
// device trains on exactly the same data.
static void generate_image_data(
    Eigen::Tensor<float, 4>& images,    // [N, C, H, W]
    Eigen::Tensor<float, 2>& labels,    // [N, num_classes]  one-hot
    int samples_per_class,
    int num_classes,
    int channels,
    int height,
    int width,
    float noise = 0.1f)
{
    const int N = samples_per_class * num_classes;
    images.resize(N, channels, height, width);
    labels.resize(N, num_classes);
    images.setZero();
    labels.setZero();

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, noise);

    int idx = 0;
    for (int cls = 0; cls < num_classes; ++cls)
    {
        // each class: a unique frequency pattern per channel
        float freq = static_cast<float>(cls + 1);
        for (int s = 0; s < samples_per_class; ++s)
        {
            for (int c = 0; c < channels; ++c)
            {
                float phase = static_cast<float>(c) * 0.5f;
                for (int h = 0; h < height; ++h)
                    for (int w = 0; w < width; ++w)
                    {
                        float val = std::sin(freq * h * 0.3f + phase)
                                  * std::cos(freq * w * 0.3f + phase);
                        images(idx, c, h, w) = val + nd(rng);
                    }
            }
            labels(idx, cls) = 1.0f;
            ++idx;
        }
    }
}


// ─── CNN configuration ───────────────────────────────────────────────
struct CNNConfig
{
    std::string label;
    // conv layers: {out_channels, kernel_size, stride, padding}
    struct ConvSpec { int out_ch; int kern; int stride; int pad; };
    std::vector<ConvSpec> convs;
    // pool after each conv: {pool_size}
    std::vector<int> pools;   // 0 = no pool after that conv
    // fc layers after flatten (excluding final output)
    std::vector<int> fc_hidden;
    int epochs;
    int batch_size;
    float lr;
};


// ─── training function ───────────────────────────────────────────────
static std::tuple<double, float, float> train_cnn(
    const CNNConfig& cfg,
    const std::string& dev,
    const Eigen::Tensor<float, 4>& images,
    const Eigen::Tensor<float, 2>& labels,
    int num_classes,
    int channels,
    int height,
    int width,
    bool verbose)
{
    const int N = images.dimension(0);

    // set OpenMP threads for CPU backends
    if (dev == "cpu" || dev == "cpu-eigen")
    {
        CppNet::Layers::Linear::set_num_threads(4);
        CppNet::Activations::ReLU::set_num_threads(4);
    }

    // ── build conv / pool / relu stack ──
    std::vector<std::shared_ptr<CppNet::Layers::Conv2D>>   conv_layers;
    std::vector<std::shared_ptr<CppNet::Layers::MaxPool2D>> pool_layers;
    std::vector<CppNet::Activations::ReLU>                  relu_layers;

    int cur_ch = channels, cur_h = height, cur_w = width;

    for (size_t i = 0; i < cfg.convs.size(); ++i)
    {
        auto& cs = cfg.convs[i];
        auto conv = std::make_shared<CppNet::Layers::Conv2D>(
            cur_ch, cs.out_ch, cs.kern, cs.stride, cs.pad, true, dev);

        #ifdef USE_CUDA
        if (dev == "gpu")
            conv->set_max_batch_size(cfg.batch_size);
        #endif

        conv_layers.push_back(conv);
        relu_layers.emplace_back(dev);

        // compute spatial dimensions after conv
        cur_h = (cur_h + 2 * cs.pad - cs.kern) / cs.stride + 1;
        cur_w = (cur_w + 2 * cs.pad - cs.kern) / cs.stride + 1;
        cur_ch = cs.out_ch;

        // pool
        if (i < cfg.pools.size() && cfg.pools[i] > 0)
        {
            auto pool = std::make_shared<CppNet::Layers::MaxPool2D>(
                cfg.pools[i], -1, dev);
            pool_layers.push_back(pool);
            cur_h = (cur_h - cfg.pools[i]) / cfg.pools[i] + 1;
            cur_w = (cur_w - cfg.pools[i]) / cfg.pools[i] + 1;
        }
        else
        {
            pool_layers.push_back(nullptr);
        }
    }

    auto flatten = std::make_shared<CppNet::Layers::Flatten>();
    int flat_size = cur_ch * cur_h * cur_w;

    // ── build FC head ──
    std::vector<std::shared_ptr<CppNet::Layers::Linear>> fc_layers;
    std::vector<CppNet::Activations::ReLU>               fc_relus;

    std::vector<int> fc_dims;
    fc_dims.push_back(flat_size);
    for (int h : cfg.fc_hidden) fc_dims.push_back(h);
    fc_dims.push_back(num_classes);

    for (int i = 0; i + 1 < (int)fc_dims.size(); ++i)
    {
        auto fc = std::make_shared<CppNet::Layers::Linear>(
            fc_dims[i], fc_dims[i + 1],
            "fc" + std::to_string(i + 1), true, true, dev, "xavier");

        #ifdef USE_CUDA
        if (dev == "gpu")
            fc->set_max_batch_size(cfg.batch_size);
        #endif

        fc_layers.push_back(fc);
        if (i + 2 < (int)fc_dims.size())
            fc_relus.emplace_back(dev);
    }

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

            // batch extraction
            Eigen::Tensor<float, 4> x_batch(bs, channels, height, width);
            Eigen::Tensor<float, 2> y_batch(bs, num_classes);
            for (int i = 0; i < bs; ++i)
            {
                int idx = indices[start + i];
                for (int c = 0; c < channels; ++c)
                    for (int h = 0; h < height; ++h)
                        for (int w = 0; w < width; ++w)
                            x_batch(i, c, h, w) = images(idx, c, h, w);
                for (int c = 0; c < num_classes; ++c)
                    y_batch(i, c) = labels(idx, c);
            }

            // ── forward ──
            Eigen::Tensor<float, 4> z = x_batch;
            for (size_t i = 0; i < conv_layers.size(); ++i)
            {
                z = conv_layers[i]->forward(z);
                z = relu_layers[i].forward(z);
                if (pool_layers[i])
                    z = pool_layers[i]->forward(z);
            }

            auto flat = flatten->forward(z);

            Eigen::Tensor<float, 2> fc_out = flat;
            for (size_t i = 0; i < fc_layers.size(); ++i)
            {
                fc_out = fc_layers[i]->forward(fc_out);
                if (i < fc_relus.size())
                    fc_out = fc_relus[i].forward(fc_out);
            }

            float batch_loss = loss.forward(fc_out, y_batch);
            epoch_loss += batch_loss;
            epoch_acc  += CppNet::Metrics::accuracy(fc_out, y_batch);
            ++num_batches;

            // ── backward ──
            auto grad2d = loss.backward(fc_out, y_batch);

            for (int i = (int)fc_layers.size() - 1; i >= 0; --i)
            {
                if (i < (int)fc_relus.size())
                    grad2d = fc_relus[i].backward(grad2d);
                grad2d = fc_layers[i]->backward(grad2d);
            }

            auto grad4d = flatten->backward(grad2d);

            for (int i = (int)conv_layers.size() - 1; i >= 0; --i)
            {
                if (pool_layers[i])
                    grad4d = pool_layers[i]->backward(grad4d);
                grad4d = relu_layers[i].backward(grad4d);
                grad4d = conv_layers[i]->backward(grad4d);
            }

            // ── update ──
            for (auto& fc : fc_layers)
                fc->step(optim, cfg.lr);
            for (auto& conv : conv_layers)
                conv->step(optim, cfg.lr);

            // reset grads
            for (auto& fc : fc_layers)
                fc->reset_grads();
            for (auto& conv : conv_layers)
                conv->reset_grads();
        }

        final_loss = epoch_loss / num_batches;
        final_acc  = epoch_acc  / num_batches;

        if (verbose)
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


// ─── architecture string helper ──────────────────────────────────────
static std::string arch_str(const CNNConfig& cfg, int in_ch)
{
    std::string s;
    int ch = in_ch;
    for (size_t i = 0; i < cfg.convs.size(); ++i)
    {
        s += "Conv(" + std::to_string(ch) + "→" + std::to_string(cfg.convs[i].out_ch)
           + ",k" + std::to_string(cfg.convs[i].kern) + ")";
        ch = cfg.convs[i].out_ch;
        if (i < cfg.pools.size() && cfg.pools[i] > 0)
            s += "→Pool" + std::to_string(cfg.pools[i]);
        s += "→";
    }
    s += "Flat";
    for (int h : cfg.fc_hidden)
        s += "→" + std::to_string(h);
    s += "→out";
    return s;
}


// ═════════════════════════════════════════════════════════════════════
int main()
{
    std::cout << "---------------------------------------------------------------\n";
    std::cout << "    CppNet — CNN Device Benchmark (Image Classification)       \n";
    std::cout << "---------------------------------------------------------------\n";

    // ── dataset parameters (CIFAR-10-like shape) ──
    const int samples_per_class = 100;
    const int num_classes        = 10;
    const int channels           = 3;
    const int height             = 32;
    const int width              = 32;
    const int N = samples_per_class * num_classes;

    Eigen::Tensor<float, 4> images;
    Eigen::Tensor<float, 2> labels;
    generate_image_data(images, labels, samples_per_class, num_classes,
                        channels, height, width, 0.1f);

    std::cout << "Dataset: " << N << " synthetic images, "
              << channels << "×" << height << "×" << width
              << ", " << num_classes << " classes\n";

    // ── CNN configurations (Small → Large) ──
    //
    //  Small:  2 conv layers (16, 32 filters), pool after each
    //  Medium: 2 conv layers (32, 64 filters), pool after each, FC hidden
    //  Large:  3 conv layers (64, 128, 128 filters), pool after 1st & 3rd

    std::vector<CNNConfig> configs = {
        {
            "Small",
            {{16, 3, 1, 1}, {32, 3, 1, 1}},
            {2, 2},
            {},           // no FC hidden, straight to output
            3, 32, 0.001f
        },
        {
            "Medium",
            {{32, 3, 1, 1}, {64, 3, 1, 1}},
            {2, 2},
            {128},        // one FC hidden
            2, 32, 0.0005f
        },
        {
            "Large",
            {{64, 3, 1, 1}, {128, 3, 1, 1}, {128, 3, 1, 1}},
            {2, 0, 2},   // pool after 1st and 3rd
            {256},
            2, 32, 0.0005f
        },
    };

    // ── devices ──
    std::vector<std::string> devices = {"cpu-eigen", "cpu"};
    #ifdef USE_CUDA
    devices.push_back("gpu");
    #endif

    // ── results storage ──
    struct Result { double ms; float loss; float acc; };
    std::map<std::string, std::map<std::string, Result>> results;

    // ── run experiments ──
    for (auto& cfg : configs)
    {
        std::string arch = arch_str(cfg, channels);
        std::cout << "\n----------------------------------------------------\n";
        std::cout << "  Config: " << cfg.label
                  << "   |   Epochs: " << cfg.epochs
                  << "   |   Batch: " << cfg.batch_size << "\n";
        std::cout << "  Arch: " << arch << "\n";
        std::cout << "----------------------------------------------------\n";

        for (auto& dev : devices)
        {
            std::cout << "\n  ▸ Device: " << dev << "\n";
            auto [ms, loss, acc] = train_cnn(cfg, dev, images, labels,
                                              num_classes, channels,
                                              height, width, true);
            results[cfg.label][dev] = {ms, loss, acc};

            std::cout << std::fixed << std::setprecision(2)
                      << "    ✓ Time: " << ms << " ms ("
                      << ms / 1000.0 << " s)   Loss: "
                      << std::setprecision(4) << loss
                      << "   Acc: " << acc << "%\n";
        }
    }

    // ── summary table ──
    std::cout << "\n\n";
    std::cout << "==================================================================\n";
    std::cout << "          CppNet CNN Device Benchmark — Summary                   \n";
    std::cout << "==================================================================\n";

    std::cout << std::left << std::setw(10) << "Config";
    for (auto& dev : devices)
        std::cout << "  " << std::setw(14) << dev;
    std::cout << "\n";
    std::cout << "------------------------------------------------------------------\n";

    for (auto& cfg : configs)
    {
        std::cout << std::left << std::setw(10) << cfg.label;
        for (auto& dev : devices)
        {
            auto& r = results[cfg.label][dev];
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << r.ms / 1000.0 << " s";
            std::cout << "  " << std::right << std::setw(14) << oss.str();
        }
        std::cout << "\n";
    }

    // speedup rows
    std::cout << "------------------------------------------------------------------\n";
    std::cout << "Speedup vs cpu-eigen:\n";
    for (auto& cfg : configs)
    {
        double baseline = results[cfg.label]["cpu-eigen"].ms;
        std::cout << std::left << std::setw(10) << cfg.label;
        for (auto& dev : devices)
        {
            if (dev == "cpu-eigen")
            {
                std::cout << "  " << std::right << std::setw(14) << "1.00x";
            }
            else
            {
                double spd = baseline / results[cfg.label][dev].ms;
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << spd << "x";
                std::cout << "  " << std::right << std::setw(14) << oss.str();
            }
        }
        std::cout << "\n";
    }

    // accuracy summary
    std::cout << "------------------------------------------------------------------\n";
    std::cout << "Final accuracy per config:\n";
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
    std::cout << "==================================================================\n";

    return 0;
}

