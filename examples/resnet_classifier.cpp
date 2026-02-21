/**
 * @file resnet_classifier.cpp
 * @brief ResNet-style classifier with skip connections for binary classification.
 *
 * Architecture:
 *   Linear(2, 32) -> ReLU
 *   -> [Linear(32,32) -> ReLU -> Linear(32,32)] + skip -> ReLU   (Residual block)
 *   -> Linear(32, 1) -> Sigmoid
 *
 * Loss:     BinaryCrossEntropy
 * Optimizer: Adam (separate per layer) with gradient clipping
 * Data:     Concentric circles (inner=class 0, outer=class 1)
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
static void generate_circles(
    Eigen::Tensor<float, 2>& X,
    Eigen::Tensor<float, 2>& Y,
    int N, float noise = 0.08f)
{
    X.resize(N, 2);
    Y.resize(N, 1);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * 3.14159265f);
    std::normal_distribution<float> noise_dist(0.0f, noise);
    int half = N / 2;
    for (int i = 0; i < half; ++i) {
        float theta = angle_dist(rng);
        float r = 0.4f + noise_dist(rng);
        X(i, 0) = r * std::cos(theta);
        X(i, 1) = r * std::sin(theta);
        Y(i, 0) = 0.0f;
    }
    for (int i = half; i < N; ++i) {
        float theta = angle_dist(rng);
        float r = 1.0f + noise_dist(rng);
        X(i, 0) = r * std::cos(theta);
        X(i, 1) = r * std::sin(theta);
        Y(i, 0) = 1.0f;
    }
}

// ---------------------------------------------------------------------------
static float compute_accuracy(const Eigen::Tensor<float, 2>& probs,
                              const Eigen::Tensor<float, 2>& targets)
{
    int n = probs.dimension(0);
    int correct = 0;
    for (int i = 0; i < n; ++i)
        if ((probs(i,0) >= 0.5f) == (targets(i,0) >= 0.5f)) ++correct;
    return static_cast<float>(correct) / n * 100.0f;
}

static Eigen::Tensor<float, 2> add_tensors(
    const Eigen::Tensor<float, 2>& a, const Eigen::Tensor<float, 2>& b)
{
    int r = a.dimension(0), c = a.dimension(1);
    Eigen::Tensor<float, 2> res(r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            res(i,j) = a(i,j) + b(i,j);
    return res;
}

// Clip gradient norms on a Linear layer's grad_weights and grad_biases
static void clip_grads(Layers::Linear& layer, float max_norm)
{
    auto& gw = layer.get_grad_weights();
    int r = gw.dimension(0), c = gw.dimension(1);
    float sq_sum = 0;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            sq_sum += gw(i,j) * gw(i,j);
    if (layer.has_bias()) {
        auto& gb = layer.get_grad_biases();
        for (int i = 0; i < gb.dimension(0); ++i)
            sq_sum += gb(i) * gb(i);
    }
    float norm = std::sqrt(sq_sum);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                gw(i,j) *= scale;
        if (layer.has_bias()) {
            auto& gb = layer.get_grad_biases();
            for (int i = 0; i < gb.dimension(0); ++i)
                gb(i) *= scale;
        }
    }
}

// ===========================================================================
int main()
{
    std::cout << "=== ResNet-Style Classifier (Concentric Circles) ===\n\n";

    const int   N     = 600;
    const int   H     = 32;
    const int   bs    = 32;
    const int   EP    = 100;
    const float lr    = 0.001f;
    const float clip  = 1.0f;

    Eigen::Tensor<float, 2> X, Y;
    generate_circles(X, Y, N);
    std::cout << "Dataset: " << N << " samples, concentric circles (inner=0, outer=1)\n";

    // ---- Layers (He init for ReLU) ----
    Layers::Linear proj(2, H, "proj", true, true, "cpu-eigen", "he");
    Layers::Linear b1a(H, H, "b1a", true, true, "cpu-eigen", "he");
    Layers::Linear b1b(H, H, "b1b", true, true, "cpu-eigen", "he");
    Layers::Linear head(H, 1, "head", true, true, "cpu-eigen", "xavier");

    // Separate activation instances (each caches own state)
    Activations::ReLU r0("cpu-eigen"), r1a("cpu-eigen"), r1post("cpu-eigen");
    Activations::Sigmoid sig;

    Losses::BinaryCrossEntropy bce("mean", false);

    // Separate Adam per layer
    Optimizers::Adam opt_proj, opt_b1a, opt_b1b, opt_head;

    struct LayerOpt { Layers::Linear* l; Optimizers::Adam* o; };
    std::vector<LayerOpt> lo = {
        {&proj, &opt_proj}, {&b1a, &opt_b1a},
        {&b1b, &opt_b1b}, {&head, &opt_head}
    };

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(99);

    std::cout << "Architecture: Linear(2," << H << ")->ReLU -> ResBlock(" << H
              << ") -> Linear(" << H << ",1)->Sigmoid\n";
    std::cout << "Optimizer: Adam (per-layer), lr=" << lr
              << ", grad_clip=" << clip << ", batch=" << bs << ", epochs=" << EP << "\n\n";

    for (int ep = 0; ep < EP; ++ep)
    {
        std::shuffle(idx.begin(), idx.end(), rng);
        float eloss = 0, eacc = 0;
        int nb = 0;

        for (int s = 0; s + bs <= N; s += bs)
        {
            // Extract batch
            Eigen::Tensor<float,2> xb(bs,2), yb(bs,1);
            for (int i = 0; i < bs; ++i) {
                xb(i,0) = X(idx[s+i],0); xb(i,1) = X(idx[s+i],1);
                yb(i,0) = Y(idx[s+i],0);
            }

            // Forward: proj -> relu -> [b1a -> relu -> b1b] + skip -> relu -> head -> sigmoid
            auto h = proj.forward(xb);
            h = r0.forward(h);

            auto skip = h;                      // save for skip connection
            auto blk = b1a.forward(h);
            blk = r1a.forward(blk);
            blk = b1b.forward(blk);
            h = add_tensors(blk, skip);         // residual add
            h = r1post.forward(h);              // post-add ReLU

            auto logits = head.forward(h);
            auto p = sig.forward(logits);

            float loss = bce.forward(p, yb);
            eloss += loss; eacc += compute_accuracy(p, yb); ++nb;

            // Backward
            auto g = bce.backward(p, yb);
            g = sig.backward(g);
            g = head.backward(g);
            g = r1post.backward(g);

            // Split gradient for skip connection
            auto g_blk = b1b.backward(g);       // block path
            g_blk = r1a.backward(g_blk);
            g_blk = b1a.backward(g_blk);
            g = add_tensors(g, g_blk);           // skip + block gradients

            g = r0.backward(g);
            proj.backward(g);

            // Clip gradients and update
            for (auto& p : lo) {
                clip_grads(*p.l, clip);
                p.l->step(*p.o, lr);
                p.l->reset_grads();
            }
        }

        if ((ep+1) % 10 == 0 || ep == 0)
            std::cout << std::fixed << std::setprecision(4)
                      << "Epoch " << std::setw(3) << ep+1
                      << "  |  Loss: " << eloss/nb
                      << "  |  Accuracy: " << eacc/nb << "%\n";
    }

    // Full-dataset evaluation
    auto h = proj.forward(X);
    h = r0.forward(h);
    auto skip = h;
    auto blk = b1a.forward(h);
    blk = r1a.forward(blk);
    blk = b1b.forward(blk);
    h = add_tensors(blk, skip);
    h = r1post.forward(h);
    auto logits = head.forward(h);
    auto probs = sig.forward(logits);

    float final_acc = compute_accuracy(probs, Y);
    std::cout << "\nFinal accuracy on full dataset: " << std::fixed << std::setprecision(2)
              << final_acc << "%\n";

    std::cout << "\nSample predictions (first 5 inner, first 5 outer):\n";
    std::cout << std::setw(8) << "X[0]" << std::setw(8) << "X[1]"
              << std::setw(8) << "Pred" << std::setw(8) << "True\n";
    int show[] = {0,1,2,3,4, 300,301,302,303,304};
    for (int i : show)
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(8) << X(i,0) << std::setw(8) << X(i,1)
                  << std::setw(8) << probs(i,0) << std::setw(8) << Y(i,0) << "\n";

    std::cout << "\n=== ResNet Example Complete ===\n";
    return 0;
}
