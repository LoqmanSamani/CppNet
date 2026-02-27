/**
 * @file sequence_benckmark.cpp
 * @brief Sequence Layer Device Benchmark — Sine-Wave Regression
 *
 * Benchmarks RNN, LSTM, and GRU layers across cpu-eigen, cpu (OpenMP),
 * and gpu (CUDA) backends using synthetic sine-wave sequences.
 *
 * Architecture per layer type:
 *   RecurrentLayer(1, H, return_sequences=true) → last hidden → Linear(H, 1)
 *
 * Loss:      MSE  (different from MLP/CNN benchmarks)
 * Optimizer: Momentum  (different from MLP/CNN benchmarks)
 * Data:      Predict next value in a discretised sine wave.
 */

#include <CppNet/CppNet.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <sstream>
#include <functional>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

// data generation 
static void generate_sine_data(
    Eigen::Tensor<float, 3>& X,   // [N, seq_len, 1]
    Eigen::Tensor<float, 2>& Y,   // [N, 1]
    int num_samples,
    int seq_len)
{
    X.resize(num_samples, seq_len, 1);
    Y.resize(num_samples, 1);
    const float step = 0.1f;
    for (int i = 0; i < num_samples; ++i)
    {
        float start = i * step;
        for (int t = 0; t < seq_len; ++t)
            X(i, t, 0) = std::sin(start + t * step);
        Y(i, 0) = std::sin(start + seq_len * step);
    }
}

// configuration struct for sequence benchmark
struct SeqConfig
{
    std::string label;
    int hidden_size;
    int seq_len;
    int epochs;
    int batch_size;
    int num_samples;
    float lr;
};

// training function
// layer_type: "RNN", "LSTM", "GRU"
static std::tuple<double, float> train_sequence(
    const SeqConfig& cfg,
    const std::string& layer_type,
    const std::string& device,
    const Eigen::Tensor<float, 3>& X,
    const Eigen::Tensor<float, 2>& Y,
    bool verbose)
{
    int N = X.dimension(0);
    int H = cfg.hidden_size;
    int seq_len = cfg.seq_len;

    if (device == "cpu" || device == "cpu-eigen")
    {
        CppNet::Layers::Linear::set_num_threads(4);
    }

    // create the recurrent layer
    std::shared_ptr<CppNet::Layers::RNN> rnn_ptr;
    std::shared_ptr<CppNet::Layers::LSTM> lstm_ptr;
    std::shared_ptr<CppNet::Layers::GRU> gru_ptr;

    if (layer_type == "RNN")
        rnn_ptr = std::make_shared<CppNet::Layers::RNN>(1, H, true, device);
    else if (layer_type == "LSTM")
        lstm_ptr = std::make_shared<CppNet::Layers::LSTM>(1, H, true, device);
    else
        gru_ptr = std::make_shared<CppNet::Layers::GRU>(1, H, true, device);

    auto linear = std::make_shared<CppNet::Layers::Linear>(
        H, 1, "output", true, true, device, "xavier");

#ifdef USE_CUDA
    if (device == "gpu")
        linear->set_max_batch_size(cfg.batch_size);
#endif

    CppNet::Losses::MSE loss;
    CppNet::Optimizers::Momentum optim(0.9f);

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    float final_loss = 0.0f;

    // lambda helpers for forward/backward/step through the recurrent layer
    auto rnn_forward = [&](const Eigen::Tensor<float, 3>& x) -> Eigen::Tensor<float, 3> {
        if (rnn_ptr)  return rnn_ptr->forward(x);
        if (lstm_ptr) return lstm_ptr->forward(x);
        return gru_ptr->forward(x);
    };
    auto rnn_backward = [&](const Eigen::Tensor<float, 3>& g) {
        if (rnn_ptr)  rnn_ptr->backward(g);
        else if (lstm_ptr) lstm_ptr->backward(g);
        else gru_ptr->backward(g);
    };
    auto rnn_step = [&](CppNet::Optimizers::Optimizer& opt, float lr) {
        if (rnn_ptr)  rnn_ptr->step(opt, lr);
        else if (lstm_ptr) lstm_ptr->step(opt, lr);
        else gru_ptr->step(opt, lr);
    };

    CppNet::Utils::Timer timer(device + "-" + layer_type, false);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int start = 0; start + cfg.batch_size <= N; start += cfg.batch_size)
        {
            int bs = cfg.batch_size;

            Eigen::Tensor<float, 3> x_batch(bs, seq_len, 1);
            Eigen::Tensor<float, 2> y_batch(bs, 1);
            for (int i = 0; i < bs; ++i)
            {
                int idx = indices[start + i];
                for (int t = 0; t < seq_len; ++t)
                    x_batch(i, t, 0) = X(idx, t, 0);
                y_batch(i, 0) = Y(idx, 0);
            }

            // forward
            auto rnn_out = rnn_forward(x_batch);

            Eigen::Tensor<float, 2> last_hidden(bs, H);
            for (int b = 0; b < bs; ++b)
                for (int d = 0; d < H; ++d)
                    last_hidden(b, d) = rnn_out(b, seq_len - 1, d);

            auto pred = linear->forward(last_hidden);
            float batch_loss = loss.forward(pred, y_batch);
            epoch_loss += batch_loss;
            ++num_batches;

            // backward
            auto grad_pred = loss.backward(pred, y_batch);
            auto grad_hidden = linear->backward(grad_pred);

            Eigen::Tensor<float, 3> grad_rnn(bs, seq_len, H);
            grad_rnn.setZero();
            for (int b = 0; b < bs; ++b)
                for (int d = 0; d < H; ++d)
                    grad_rnn(b, seq_len - 1, d) = grad_hidden(b, d);

            rnn_backward(grad_rnn);

            // update
            rnn_step(optim, cfg.lr);
            linear->step(optim, cfg.lr);
            linear->reset_grads();
        }

        final_loss = (num_batches > 0) ? epoch_loss / num_batches : 0.0f;

        if (verbose && ((epoch + 1) % std::max(1, cfg.epochs / 5) == 0))
        {
            std::cout << "  Epoch " << std::setw(3) << epoch + 1
                      << "  Loss: " << std::fixed << std::setprecision(6)
                      << final_loss << "\n";
        }
    }

    double elapsed = timer.elapsed_seconds();
    return {elapsed, final_loss};
}

 
int main()
{
    std::cout << "---------------------------------------------------------------\n"
              << "-     Sequence Layer Device Benchmark (MSE + Momentum)        -\n"
              << "---------------------------------------------------------------\n\n";

    std::vector<SeqConfig> configs = {
        {"Small",  64,  20, 5, 32, 800, 0.01f},
        {"Medium", 128, 30, 3, 32, 800, 0.005f},
        {"Large",  256, 50, 3, 32, 1200, 0.002f},
    };

    std::vector<std::string> layer_types = {"RNN", "LSTM", "GRU"};

    std::vector<std::string> devices = {
        "cpu-eigen",
        "cpu",
#ifdef USE_CUDA
        "gpu",
#endif
    };

    // results storage
    struct Result
    {
        std::string config;
        std::string layer;
        std::string device;
        double time_s;
        float loss;
    };
    std::vector<Result> results;

    for (auto& cfg : configs)
    {
        std::cout << "-------------------------------------------------------------\n";
        std::cout << "Config: " << cfg.label
                  << "  (H=" << cfg.hidden_size
                  << ", seq=" << cfg.seq_len
                  << ", epochs=" << cfg.epochs
                  << ", batch=" << cfg.batch_size
                  << ", N=" << cfg.num_samples << ")\n";
        std::cout << "-------------------------------------------------------------\n";

        Eigen::Tensor<float, 3> X;
        Eigen::Tensor<float, 2> Y;
        generate_sine_data(X, Y, cfg.num_samples, cfg.seq_len);

        for (auto& lt : layer_types)
        {
            std::cout << "\n── " << lt << " ──\n";
            for (auto& dev : devices)
            {
                std::cout << "  [" << dev << "] ";
                std::cout.flush();

                auto [t, l] = train_sequence(cfg, lt, dev, X, Y, false);
                std::cout << std::fixed << std::setprecision(2)
                          << t << " s"
                          << "  (loss=" << std::setprecision(6) << l << ")\n";

                results.push_back({cfg.label, lt, dev, t, l});
            }
        }
        std::cout << "\n";
    }

    // summary table
    std::cout << "\n-------------------------------------------------------------\n"
              << "-                        Summary                              -\n"
              << "---------------------------------------------------------------\n\n";

    std::cout << std::left
              << std::setw(8) << "Config"
              << std::setw(7) << "Layer"
              << std::setw(12) << "Device"
              << std::right
              << std::setw(10) << "Time (s)"
              << std::setw(14) << "Final Loss"
              << "\n";
    std::cout << std::string(51, '-') << "\n";

    for (auto& r : results)
    {
        std::cout << std::left
                  << std::setw(8) << r.config
                  << std::setw(7) << r.layer
                  << std::setw(12) << r.device
                  << std::right << std::fixed
                  << std::setw(10) << std::setprecision(2) << r.time_s
                  << std::setw(14) << std::setprecision(6) << r.loss
                  << "\n";
    }

    // speedup analysis
    std::cout << "\n-  Speedup vs cpu-eigen  -\n";
    for (auto& cfg : configs)
    {
        for (auto& lt : layer_types)
        {
            double eigen_time = 0.0;
            for (auto& r : results)
                if (r.config == cfg.label && r.layer == lt && r.device == "cpu-eigen")
                    eigen_time = r.time_s;
            if (eigen_time <= 0) continue;

            std::cout << cfg.label << "/" << lt << ":  ";
            for (auto& r : results)
            {
                if (r.config == cfg.label && r.layer == lt && r.device != "cpu-eigen")
                {
                    double speedup = eigen_time / r.time_s;
                    std::cout << r.device << "=" << std::fixed
                              << std::setprecision(1) << speedup << "x  ";
                }
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n-  Sequence Benchmark Complete  -\n";
    return 0;
}
