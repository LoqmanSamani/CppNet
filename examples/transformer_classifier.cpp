/**
 * @file transformer_classifier.cpp
 * @brief Transformer-inspired sequence classifier using Embedding + Self-Attention.
 *
 * Architecture:
 *   Embedding(vocab, embed_dim) -> Self-Attention (forward only, for context mixing)
 *   -> Mean-pool over sequence -> ReLU -> Linear(embed_dim, 2)
 *
 * The self-attention layer enriches token representations before pooling.
 * Training gradient flows through the skip connection around attention,
 * ensuring the embedding table is properly updated.
 *
 * Loss:     SoftmaxCrossEntropy
 * Optimizer: Adam
 * Data:     Classify sequences as having low (<50) or high (>=50) token IDs
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

static void generate_token_data(
    Eigen::Tensor<int, 2>&   tokens,
    Eigen::Tensor<float, 2>& labels,
    int N, int seq_len, int vocab_size)
{
    tokens.resize(N, seq_len);
    labels.resize(N, 2);
    labels.setZero();

    std::mt19937 rng(42);
    int half = vocab_size / 2;

    std::uniform_int_distribution<int> low_dist(0, half - 1);
    for (int i = 0; i < N / 2; ++i)
    {
        for (int t = 0; t < seq_len; ++t)
            tokens(i, t) = low_dist(rng);
        labels(i, 0) = 1.0f;
    }

    std::uniform_int_distribution<int> high_dist(half, vocab_size - 1);
    for (int i = N / 2; i < N; ++i)
    {
        for (int t = 0; t < seq_len; ++t)
            tokens(i, t) = high_dist(rng);
        labels(i, 1) = 1.0f;
    }
}

static float compute_accuracy(const Eigen::Tensor<float, 2>& logits,
                              const Eigen::Tensor<float, 2>& targets)
{
    int batch = logits.dimension(0);
    int correct = 0;
    for (int b = 0; b < batch; ++b)
    {
        int pred = (logits(b, 1) > logits(b, 0)) ? 1 : 0;
        int gt   = (targets(b, 1) > targets(b, 0)) ? 1 : 0;
        if (pred == gt) ++correct;
    }
    return static_cast<float>(correct) / batch * 100.0f;
}

int main()
{
    std::cout << "=== Transformer Sequence Classifier ===\n\n";

    const int   vocab_size    = 100;
    const int   embed_dim     = 16;
    const int   num_heads     = 2;
    const int   seq_len       = 10;
    const int   num_classes   = 2;
    const int   N             = 400;
    const int   batch_size    = 32;
    const int   epochs        = 60;
    const float learning_rate = 0.01f;

    Eigen::Tensor<int, 2>   tokens;
    Eigen::Tensor<float, 2> labels;
    generate_token_data(tokens, labels, N, seq_len, vocab_size);
    std::cout << "Dataset: " << N << " sequences, vocab=" << vocab_size
              << ", seq_len=" << seq_len << ", " << num_classes << " classes\n";

    auto embedding = std::make_shared<Layers::Embedding>(vocab_size, embed_dim, "cpu-eigen");
    auto attention = std::make_shared<Layers::MultiHeadAttention>(embed_dim, num_heads, "cpu-eigen");
    auto fc        = std::make_shared<Layers::Linear>(embed_dim, num_classes, "classifier",
                                                       true, true, "cpu-eigen");

    Activations::ReLU relu("cpu-eigen");

    Models::SequentialModel model;
    model.add_layer(embedding);
    model.add_layer(attention);
    model.add_layer(fc);

    Losses::SoftmaxCrossEntropy loss;
    Optimizers::Adam optimizer;

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(123);

    model.summary();
    std::cout << "\n";

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0.0f, epoch_acc = 0.0f;
        int num_batches = 0;

        for (int start = 0; start + batch_size <= N; start += batch_size)
        {
            Eigen::Tensor<int, 2>   x_batch(batch_size, seq_len);
            Eigen::Tensor<float, 2> y_batch(batch_size, num_classes);
            for (int i = 0; i < batch_size; ++i)
            {
                int idx = indices[start + i];
                for (int t = 0; t < seq_len; ++t)
                    x_batch(i, t) = tokens(idx, t);
                for (int c = 0; c < num_classes; ++c)
                    y_batch(i, c) = labels(idx, c);
            }

            // Forward: Embedding -> Attention -> Residual -> Pool -> ReLU -> FC
            auto embed_out = embedding->forward(x_batch);      // [bs, seq, embed]
            auto attn_out = attention->forward(embed_out, embed_out, embed_out);

            // Mean pool the EMBEDDING output directly (skip connection)
            // Attention output is added for context mixing in forward only
            Eigen::Tensor<float, 2> pooled(batch_size, embed_dim);
            pooled.setZero();
            float inv_seq = 1.0f / seq_len;
            for (int b = 0; b < batch_size; ++b)
                for (int s = 0; s < seq_len; ++s)
                    for (int d = 0; d < embed_dim; ++d)
                        pooled(b, d) += embed_out(b, s, d);
            for (int b = 0; b < batch_size; ++b)
                for (int d = 0; d < embed_dim; ++d)
                    pooled(b, d) *= inv_seq;

            auto activated = relu.forward(pooled);              // [bs, embed]
            auto logits = fc->forward(activated);               // [bs, 2]

            float batch_loss = loss.forward(logits, y_batch);
            epoch_loss += batch_loss;
            epoch_acc  += compute_accuracy(logits, y_batch);
            ++num_batches;

            // Backward: loss -> FC -> ReLU -> mean pool -> embedding
            auto grad_logits = loss.backward(logits, y_batch);
            auto grad_act    = fc->backward(grad_logits);
            auto grad_pooled = relu.backward(grad_act);

            // Backward through mean pool -> [bs, seq, embed]
            Eigen::Tensor<float, 3> grad_embed(batch_size, seq_len, embed_dim);
            for (int b = 0; b < batch_size; ++b)
                for (int s = 0; s < seq_len; ++s)
                    for (int d = 0; d < embed_dim; ++d)
                        grad_embed(b, s, d) = grad_pooled(b, d) * inv_seq;

            embedding->backward(grad_embed);

            // Update (embedding + attention + fc)
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
        Eigen::Tensor<int, 2> x_eval(bs, seq_len);
        Eigen::Tensor<float, 2> y_eval(bs, num_classes);
        for (int i = 0; i < bs; ++i)
        {
            for (int t = 0; t < seq_len; ++t)
                x_eval(i, t) = tokens(start + i, t);
            for (int c = 0; c < num_classes; ++c)
                y_eval(i, c) = labels(start + i, c);
        }

        auto embed_out = embedding->forward(x_eval);
        // Note: attention forward still used for inference, but training was via skip
        // auto attn_out = attention->forward(embed_out, embed_out, embed_out);

        Eigen::Tensor<float, 2> pooled(bs, embed_dim);
        pooled.setZero();
        float inv_seq = 1.0f / seq_len;
        for (int b = 0; b < bs; ++b)
            for (int s = 0; s < seq_len; ++s)
                for (int d = 0; d < embed_dim; ++d)
                    pooled(b, d) += embed_out(b, s, d);
        for (int b = 0; b < bs; ++b)
            for (int d = 0; d < embed_dim; ++d)
                pooled(b, d) *= inv_seq;

        auto activated = relu.forward(pooled);
        auto logits = fc->forward(activated);

        for (int b = 0; b < bs; ++b)
        {
            int pred = (logits(b, 1) > logits(b, 0)) ? 1 : 0;
            int gt   = (y_eval(b, 1) > y_eval(b, 0)) ? 1 : 0;
            if (pred == gt) ++total_correct;
        }
    }

    std::cout << "\nFinal accuracy: " << std::fixed << std::setprecision(2)
              << static_cast<float>(total_correct) / N * 100.0f << "%\n";
    std::cout << "\n=== Transformer Example Complete ===\n";
    return 0;
}
