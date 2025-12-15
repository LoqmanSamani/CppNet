#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h>

#include "CppNet/layers/linear.hpp"
#include "CppNet/optimizers/sgd.hpp"
#include "CppNet/activations/relu.hpp"
#include "CppNet/activations/sigmoid.hpp"
#include "CppNet/losses/binary_cross_entropy.hpp"




class DataProcessor {
    public:
        DataProcessor(bool has_header = true) : header(has_header) {}

        Eigen::MatrixXf load_data(const std::string& path, int num_feats, float proportion = 1.0f) {
            std::ifstream data(path);
            if (!data.is_open()) {
                throw std::runtime_error("Could not open the data!");
            }

            // count number of rows
            int row_count = 0;
            std::string line;
            while (std::getline(data, line)) {
                if (!line.empty()) {
                    row_count++;
                }
            }

            // reset file stream
            data.clear();
            data.seekg(0);

            if (header) {
                std::getline(data, line); // skip header
                row_count--;
            }

            int max_rows = static_cast<int>(row_count * proportion);
            Eigen::MatrixXf data_array(max_rows, num_feats);

            int j = 0;
            while (j < max_rows && std::getline(data, line)) {
                if (line.empty()) continue;

                std::vector<float> feats;
                std::stringstream ss(line);
                std::string val;

                while (std::getline(ss, val, ',')) {
                    feats.push_back(std::stod(val));
                }

                if (feats.size() >= static_cast<size_t>(num_feats)) {
                    for (int i = 0; i < num_feats; ++i) {
                        data_array(j, i) = feats[i];
                    }
                    j++;
                }
            }

            return data_array.topRows(j); // in case fewer rows were read
        }

        void split_data(
            const Eigen::MatrixXf& data,
            Eigen::MatrixXf& train_data,
            Eigen::MatrixXf& test_data,
            Eigen::MatrixXf& val_data,
            bool use_val = false) const {
            
            // check if total rows match
            if (!use_val) {
                if (data.rows() != train_data.rows() + test_data.rows()) {
                    throw std::runtime_error("Shape Mismatch!");
                }
            } else {
                if (data.rows() != train_data.rows() + test_data.rows() + val_data.rows()) {
                    throw std::runtime_error("Shape Mismatch!");
                }
            }
            
            // copy training data
            for (int i = 0; i < train_data.rows(); i++) {
                for (int j = 0; j < data.cols(); j++) {
                    train_data(i, j) = data(i, j);
                }
            }
            
            if (use_val) {
                // copy validation data
                for (int i = 0; i < val_data.rows(); i++) {
                    for (int j = 0; j < data.cols(); j++) {
                        val_data(i, j) = data(train_data.rows() + i, j);
                    }
                }
                
                // copy test data
                for (int i = 0; i < test_data.rows(); i++) {
                    for (int j = 0; j < data.cols(); j++) {
                        test_data(i, j) = data(train_data.rows() + val_data.rows() + i, j);
                    }
                }
            } else {
                // copy test data when no validation set
                for (int i = 0; i < test_data.rows(); i++) {
                    for (int j = 0; j < data.cols(); j++) {
                        test_data(i, j) = data(train_data.rows() + i, j);
                    }
                }
            }
        }

        void prepare_batch(const Eigen::MatrixXf& data, Eigen::MatrixXf& X_batch, Eigen::MatrixXf& y_batch, std::vector<int>& indices) const {
            for (int i = 0; i < X_batch.rows(); i++) {
                for (int j = 0; j < X_batch.cols(); j++) {
                    X_batch(i, j) = data(indices[i], j);
                }
                y_batch(i, 0) = data(indices[i], data.cols()-1);
            }
        }

    private:
        bool header;
};

void standardize(Eigen::MatrixXf& data) {
    for (Eigen::Index j = 0; j < data.cols()-1; ++j) {
        float mean = data.col(j).mean();
        float std = std::sqrt((data.col(j).array() - mean).square().mean());
        if (std > 0) {
            data.col(j) = (data.col(j).array() - mean) / std;
        } else {
            data.col(j).setZero();
        }
    }
}

float run_training_with_threads(
    int num_threads,
    int epochs_to_test,
    const std::string& device)
{
    std::cout << "\n=== Testing with " << num_threads
              << " threads (" << device << ") ===\n";

    DataProcessor dp(true);
    Eigen::MatrixXf data = dp.load_data("../examples/breast_cancer.csv", 31, 1.0);

    Eigen::MatrixXf train_data(500, 31);
    Eigen::MatrixXf test_data(39, 31);
    Eigen::MatrixXf val_data(30, 31);
    dp.split_data(data, train_data, test_data, val_data, true);

    // ---------------- Layers ----------------
    CppNet::Layers::Linear layer1(30, 50, "TestLayer1", true, true, device, "he");
    CppNet::Activations::ReLU relu1(device);

    CppNet::Layers::Linear layer2(50, 100, "TestLayer2", true, true, device, "he");
    CppNet::Activations::ReLU relu2(device);

    CppNet::Layers::Linear layer3(100, 100, "TestLayer3", true, true, device, "he");
    CppNet::Activations::ReLU relu3(device);

    CppNet::Layers::Linear layer4(100, 50, "TestLayer4", true, true, device, "he");
    CppNet::Activations::ReLU relu4(device);

    CppNet::Layers::Linear layer5(50, 30, "TestLayer5", true, true, device, "he");
    CppNet::Activations::ReLU relu5(device);

    CppNet::Layers::Linear layer6(30, 1, "TestLayer6", true, true, device, "he");
    CppNet::Activations::Sigmoid sigmoid;

    CppNet::Optimizers::SGD optimizer;
    CppNet::Losses::BinaryCrossEntropy loss_fn("mean", false, 1.0f);

    // ---------------- Threads ----------------
    layer1.set_num_threads(num_threads);
    layer2.set_num_threads(num_threads);
    layer3.set_num_threads(num_threads);
    layer4.set_num_threads(num_threads);
    layer5.set_num_threads(num_threads);
    layer6.set_num_threads(num_threads);

    // ---------------- GPU INIT (ONLY IF GPU) ----------------
    if (device == "gpu") {
        constexpr int MAX_BATCH = 128;
        layer1.set_max_batch_size(MAX_BATCH);
        layer2.set_max_batch_size(MAX_BATCH);
        layer3.set_max_batch_size(MAX_BATCH);
        layer4.set_max_batch_size(MAX_BATCH);
        layer5.set_max_batch_size(MAX_BATCH);
        layer6.set_max_batch_size(MAX_BATCH);
    }

    // ---------------- Training Loop ----------------
    const int batch_size = 64;
    const int iters =
        (train_data.rows() + batch_size - 1) / batch_size;

    std::vector<int> indices(train_data.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(3);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs_to_test; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;

        for (int it = 0; it < iters; ++it) {
            int start = it * batch_size;
            int end = std::min(start + batch_size,
                               static_cast<int>(train_data.rows()));
            int bs = end - start;

            Eigen::MatrixXf x(bs, train_data.cols() - 1);
            Eigen::MatrixXf y(bs, 1);

            std::vector<int> batch_idx(indices.begin() + start,
                                       indices.begin() + end);

            dp.prepare_batch(train_data, x, y, batch_idx);
            standardize(x);

            Eigen::TensorMap<Eigen::Tensor<float,2>> xt(x.data(), x.rows(), x.cols());
            Eigen::TensorMap<Eigen::Tensor<float,2>> yt(y.data(), y.rows(), y.cols());

            auto o1 = relu1.forward(layer1.forward(xt));
            auto o2 = relu2.forward(layer2.forward(o1));
            auto o3 = relu3.forward(layer3.forward(o2));
            auto o4 = relu4.forward(layer4.forward(o3));
            auto o5 = relu5.forward(layer5.forward(o4));
            auto out = sigmoid.forward(layer6.forward(o5));

            float loss = loss_fn.forward(out, yt);
            epoch_loss += loss;

            Eigen::Map<Eigen::MatrixXf> pm(out.data(), out.dimension(0), 1);
            Eigen::Map<Eigen::MatrixXf> ym(yt.data(), yt.dimension(0), 1);
            epoch_acc += (pm.array() > 0.5).cast<float>()
                         .cwiseEqual(ym.array()).mean();

            layer1.reset_grads();
            layer2.reset_grads();
            layer3.reset_grads();
            layer4.reset_grads();
            layer5.reset_grads();
            layer6.reset_grads();

            auto g = loss_fn.backward(out, yt);
            g = layer6.backward(sigmoid.backward(g));
            g = layer5.backward(relu5.backward(g));
            g = layer4.backward(relu4.backward(g));
            g = layer3.backward(relu3.backward(g));
            g = layer2.backward(relu2.backward(g));
            layer1.backward(relu1.backward(g));

            layer1.step(optimizer, 0.001f);
            layer2.step(optimizer, 0.001f);
            layer3.step(optimizer, 0.001f);
            layer4.step(optimizer, 0.001f);
            layer5.step(optimizer, 0.001f);
            layer6.step(optimizer, 0.001f);

            if (device == "gpu") {
                layer1.sync_weights_to_gpu();
                layer2.sync_weights_to_gpu();
                layer3.sync_weights_to_gpu();
                layer4.sync_weights_to_gpu();
                layer5.sync_weights_to_gpu();
                layer6.sync_weights_to_gpu();
            }
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Loss " << epoch_loss / iters
                      << " | Acc " << epoch_acc / iters << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float sec = std::chrono::duration<float>(t1 - t0).count();

    std::cout << "Training completed in " << sec << " seconds\n";
    return sec;
}

int main() {
    std::cout << "=== Deep Learning Library Performance Test ===\n";
    std::cout << "Testing with different numbers of CPU threads\n";
    std::cout << "My system has 8 CPU cores available\n";

    std::vector<int> thread_counts = {4};
    int test_epochs = 1000;

    std::vector<std::string> devices = {"cpu-eigen", "cpu", "gpu"};

    for (size_t i = 0; i < devices.size(); ++i) {
        std::vector<std::pair<int, float>> results;
        std::cout << "\n=== DEVICE: " << devices[i] << " ===\n";

        for (int threads : thread_counts) {
            float time_taken =
                run_training_with_threads(threads, test_epochs, devices[i]);

            results.push_back({threads, time_taken});

            std::cout << "Waiting 2 seconds before next test...\n\n";
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }

        std::cout << "\n=== PERFORMANCE COMPARISON RESULTS ===\n";
        std::cout << "Threads | Time (s) | Speedup | Efficiency\n";
        std::cout << "--------|----------|---------|-----------\n";

        float baseline_time = results[0].second;

        for (const auto& r : results) {
            float speedup = baseline_time / r.second;
            float efficiency = (speedup / r.first) * 100.0f;

            std::cout << r.first << "\t | "
                      << std::fixed << std::setprecision(2) << r.second
                      << "\t | " << speedup
                      << "x\t | " << efficiency << "%\n";
        }
    }

    return 0;
}
