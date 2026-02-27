/**
 * @file transformer_benckmark.cpp
 * @brief Transformer Device Benchmark — Token Classification
 *
 * Benchmarks a Transformer-based classifier across cpu-eigen, cpu (OpenMP),
 * and gpu (CUDA) backends using synthetic token sequences.
 *
 * Architecture per config:
 *   Embedding(vocab, embed_dim) → MultiHeadAttention(embed_dim, num_heads)
 *   → Mean Pool → ReLU → Linear(embed_dim, num_classes)
 *
 * Loss:      SoftmaxCrossEntropy
 * Optimizer: Adam (β₁=0.9, β₂=0.999, ε=1e-10)
 * Data:      Classify sequences by token range (low vs high IDs)
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
#include <map>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

// ─── Data Generation ─────────────────────────────────────────────────────────
static void generate_token_data(
    Eigen::Tensor<int, 2>&   tokens,
    Eigen::Tensor<float, 2>& labels,
    int N, int seq_len, int vocab_size, int num_classes)
{
    tokens.resize(N, seq_len);
    labels.resize(N, num_classes);
    labels.setZero();

    std::mt19937 rng(42);
    int samples_per_class = N / num_classes;

    for (int c = 0; c < num_classes; ++c)
    {
        int lo = c * (vocab_size / num_classes);
        int hi = (c + 1) * (vocab_size / num_classes) - 1;
        std::uniform_int_distribution<int> dist(lo, hi);

        for (int i = 0; i < samples_per_class; ++i)
        {
            int idx = c * samples_per_class + i;
            for (int t = 0; t < seq_len; ++t)
                tokens(idx, t) = dist(rng);
            labels(idx, c) = 1.0f;
        }
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────
struct TransformerConfig
{
    std::string label;
    int vocab_size;
    int embed_dim;
    int num_heads;
    int seq_len;
    int num_classes;
    int epochs;
    int batch_size;
    int num_samples;
    float lr;
};

// ─── Training Function ──────────────────────────────────────────────────────
static std::tuple<double, float, float> train_transformer(
    const TransformerConfig& cfg,
    const std::string& device,
    const Eigen::Tensor<int, 2>& tokens,
    const Eigen::Tensor<float, 2>& labels,
    bool verbose)
{
    int N = tokens.dimension(0);
    int S = cfg.seq_len;
    int D = cfg.embed_dim;
    int C = cfg.num_classes;

    if (device == "cpu" || device == "cpu-eigen")
    {
        CppNet::Layers::Linear::set_num_threads(4);
        CppNet::Activations::ReLU::set_num_threads(4);
    }

    // Build transformer model
    auto embedding = std::make_shared<CppNet::Layers::Embedding>(
        cfg.vocab_size, D, device);
    auto attention = std::make_shared<CppNet::Layers::MultiHeadAttention>(
        D, cfg.num_heads, device);
    auto fc = std::make_shared<CppNet::Layers::Linear>(
        D, C, "classifier", true, true, device, "xavier");

    CppNet::Activations::ReLU relu(device);

#ifdef USE_CUDA
    if (device == "gpu")
        fc->set_max_batch_size(cfg.batch_size);
#endif

    CppNet::Losses::SoftmaxCrossEntropy loss("mean");
    CppNet::Optimizers::Adam optim(0.9f, 0.999f, 1e-10f);

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123);

    float final_loss = 0.0f, final_acc = 0.0f;

    CppNet::Utils::Timer timer(device, false);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0.0f, epoch_acc = 0.0f;
        int num_batches = 0;

        for (int start = 0; start + cfg.batch_size <= N; start += cfg.batch_size)
        {
            int bs = cfg.batch_size;

            // Prepare batch
            Eigen::Tensor<int, 2>   x_batch(bs, S);
            Eigen::Tensor<float, 2> y_batch(bs, C);
            for (int i = 0; i < bs; ++i)
            {
                int idx = indices[start + i];
                for (int t = 0; t < S; ++t)
                    x_batch(i, t) = tokens(idx, t);
                for (int c = 0; c < C; ++c)
                    y_batch(i, c) = labels(idx, c);
            }

            // Forward: Embedding -> Attention -> Mean Pool -> ReLU -> FC
            auto embed_out = embedding->forward(x_batch);       // [bs, S, D]
            auto attn_out = attention->forward(embed_out, embed_out, embed_out); // [bs, S, D]

            // Mean pool over sequence dimension
            Eigen::Tensor<float, 2> pooled(bs, D);
            pooled.setZero();
            float inv_seq = 1.0f / S;
            for (int b = 0; b < bs; ++b)
                for (int s = 0; s < S; ++s)
                    for (int d = 0; d < D; ++d)
                        pooled(b, d) += attn_out(b, s, d);
            for (int b = 0; b < bs; ++b)
                for (int d = 0; d < D; ++d)
                    pooled(b, d) *= inv_seq;

            auto activated = relu.forward(pooled);               // [bs, D]
            auto logits = fc->forward(activated);                // [bs, C]

            float batch_loss = loss.forward(logits, y_batch);
            epoch_loss += batch_loss;
            epoch_acc  += CppNet::Metrics::accuracy(logits, y_batch);
            ++num_batches;

            // Backward
            auto grad_logits = loss.backward(logits, y_batch);
            auto grad_act    = fc->backward(grad_logits);
            auto grad_pooled = relu.backward(grad_act);

            // Backward through mean pool
            Eigen::Tensor<float, 3> grad_attn(bs, S, D);
            for (int b = 0; b < bs; ++b)
                for (int s = 0; s < S; ++s)
                    for (int d = 0; d < D; ++d)
                        grad_attn(b, s, d) = grad_pooled(b, d) * inv_seq;

            // Backward through attention
            auto grad_embed = attention->backward(grad_attn);

            // Backward through embedding
            embedding->backward(grad_embed);

            // Update all parameters
            embedding->step(optim, cfg.lr);
            attention->step(optim, cfg.lr);
            fc->step(optim, cfg.lr);
            fc->reset_grads();
        }

        final_loss = (num_batches > 0) ? epoch_loss / num_batches : 0.0f;
        final_acc  = (num_batches > 0) ? epoch_acc  / num_batches : 0.0f;

        if (verbose && ((epoch + 1) % std::max(1, cfg.epochs / 5) == 0 || epoch == 0))
        {
            std::cout << std::fixed << std::setprecision(4)
                      << "    Epoch " << std::setw(3) << epoch + 1
                      << "  |  Loss: " << final_loss
                      << "  |  Accuracy: " << final_acc << "%\n";
        }
    }

    double elapsed = timer.elapsed_ms();
    return {elapsed, final_loss, final_acc};
}

// ─── Architecture description ────────────────────────────────────────────────
static std::string arch_str(const TransformerConfig& cfg)
{
    std::ostringstream oss;
    oss << "Emb(" << cfg.vocab_size << "," << cfg.embed_dim << ")"
        << "→Attn(h=" << cfg.num_heads << ")"
        << "→Pool→FC(" << cfg.embed_dim << "," << cfg.num_classes << ")";
    return oss.str();
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main()
{
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n"
              << "║     CppNet — Transformer Device Benchmark (Token Classifier)   ║\n"
              << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // Configurations from small to large
    std::vector<TransformerConfig> configs = {
        //         label   vocab  emb  heads seq  cls  epochs batch   N     lr
        {"Small",    200,   32,   2,   10,   4,   10,   32,   800,  0.005f},
        {"Medium",   500,   64,   4,   20,   4,   5,    32,   800,  0.003f},
        {"Large",    1000,  128,  8,   30,   4,   3,    32,   1200, 0.001f},
    };

    // Devices to benchmark
    std::vector<std::string> devices = {"cpu-eigen", "cpu"};
#ifdef USE_CUDA
    devices.push_back("gpu");
#endif

    // Results storage
    struct Result
    {
        double ms;
        float loss;
        float acc;
    };
    std::map<std::string, std::map<std::string, Result>> results;

    // Run all experiments
    for (auto& cfg : configs)
    {
        std::string arch = arch_str(cfg);

        Eigen::Tensor<int, 2>   tokens;
        Eigen::Tensor<float, 2> labels;
        generate_token_data(tokens, labels, cfg.num_samples, cfg.seq_len,
                            cfg.vocab_size, cfg.num_classes);

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Config: " << cfg.label << "   |   " << arch << "\n";
        std::cout << "  vocab=" << cfg.vocab_size
                  << ", embed=" << cfg.embed_dim
                  << ", heads=" << cfg.num_heads
                  << ", seq=" << cfg.seq_len
                  << ", classes=" << cfg.num_classes
                  << ", epochs=" << cfg.epochs
                  << ", batch=" << cfg.batch_size
                  << ", N=" << cfg.num_samples << "\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

        for (auto& dev : devices)
        {
            std::cout << "\n  ▸ Device: " << dev << "\n";
            auto [ms, loss, acc] = train_transformer(cfg, dev, tokens, labels, true);
            results[cfg.label][dev] = {ms, loss, acc};

            std::cout << std::fixed << std::setprecision(2)
                      << "    ✓ Time: " << ms << " ms ("
                      << ms / 1000.0 << " s)   Loss: "
                      << std::setprecision(4) << loss
                      << "   Acc: " << acc << "%\n";
        }
    }

    // ── Summary Table ──
    std::cout << "\n\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n"
              << "║                 Transformer Benchmark — Summary                ║\n"
              << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // Timing table
    std::cout << std::left
              << std::setw(10) << "Config"
              << std::setw(45) << "Architecture";
    for (auto& dev : devices)
        std::cout << std::setw(16) << dev;
    std::cout << "\n" << std::string(10 + 45 + 16 * devices.size(), '-') << "\n";

    for (auto& cfg : configs)
    {
        std::string arch = arch_str(cfg);
        std::cout << std::left << std::setw(10) << cfg.label
                  << std::setw(45) << arch;

        for (auto& dev : devices)
        {
            auto& r = results[cfg.label][dev];
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << r.ms / 1000.0 << " s";
            std::cout << std::right << std::setw(14) << oss.str() << "  ";
        }
        std::cout << "\n";
    }

    // Speedup rows
    std::cout << "\n── Speedup vs cpu-eigen ──\n";
    for (auto& cfg : configs)
    {
        double baseline = results[cfg.label]["cpu-eigen"].ms;
        std::cout << "  " << std::left << std::setw(10) << cfg.label << ":  ";

        for (auto& dev : devices)
        {
            if (dev == "cpu-eigen")
            {
                std::cout << dev << "=1.00x  ";
            }
            else
            {
                double speedup = baseline / results[cfg.label][dev].ms;
                std::cout << dev << "=" << std::fixed << std::setprecision(2)
                          << speedup << "x  ";
            }
        }
        std::cout << "\n";
    }

    // Accuracy summary
    std::cout << "\n── Final Accuracy ──\n";
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

    // Loss summary
    std::cout << "\n── Final Loss ──\n";
    for (auto& cfg : configs)
    {
        std::cout << "  " << std::left << std::setw(10) << cfg.label << ":  ";
        for (auto& dev : devices)
        {
            std::cout << dev << "=" << std::fixed << std::setprecision(4)
                      << results[cfg.label][dev].loss << "  ";
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Transformer Benchmark Complete ===\n";
    return 0;
}
