
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
#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"
#include "losses.hpp"


class DataProcessor {
    public:
        DataProcessor(bool has_header = true) : header(has_header) {}

        Eigen::MatrixXd load_data(const std::string& path, int num_feats, double proportion = 1.0) {
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
            Eigen::MatrixXd data_array(max_rows, num_feats);

            int j = 0;
            while (j < max_rows && std::getline(data, line)) {
                if (line.empty()) continue;

                std::vector<double> feats;
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
            const Eigen::MatrixXd& data,
            Eigen::MatrixXd& train_data,
            Eigen::MatrixXd& test_data,
            Eigen::MatrixXd& val_data,
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

        void prepare_batch(const Eigen::MatrixXd& data, Eigen::MatrixXd& X_batch, Eigen::MatrixXd& y_batch, std::vector<int>& indices) const {
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

void standardize(Eigen::MatrixXd& data) {
    for (Eigen::Index j = 0; j < data.cols()-1; ++j) {
        double mean = data.col(j).mean();
        double std = std::sqrt((data.col(j).array() - mean).square().mean());
        if (std > 0) {
            data.col(j) = (data.col(j).array() - mean) / std;
        } else {
            data.col(j).setZero();
        }
    }
}

// Function to run training with specific number of threads and return timing results
double run_training_with_threads(int num_threads, int epochs_to_test = 50) {
    std::cout << "\n=== Testing with " << num_threads << " threads ===" << std::endl;
    
    // Load and prepare data
    DataProcessor dp(true);
    Eigen::MatrixXd data = dp.load_data("../examples/breast_cancer.csv", 31, 1.0);
    Eigen::MatrixXd train_data(500, 31);
    Eigen::MatrixXd test_data(39, 31);
    Eigen::MatrixXd val_data(30, 31);
    dp.split_data(data, train_data, test_data, val_data, true);

    // Create model layers
    CppNet::Layers::Linear layer1(30, 50, "TestLayer1", true, true, "cpu", "xavier");
    CppNet::Activations::ReLU relu1;
    CppNet::Layers::Linear layer2(50, 100, "TestLayer2", true, true, "cpu", "xavier");
    CppNet::Activations::ReLU relu2;
    CppNet::Layers::Linear layer3(100, 100, "TestLayer3", true, true, "cpu", "xavier");
    CppNet::Activations::ReLU relu3;
    CppNet::Layers::Linear layer4(100, 50, "TestLayer4", true, true, "cpu", "xavier");
    CppNet::Activations::ReLU relu4;
    CppNet::Layers::Linear layer5(50, 30, "TestLayer5", true, true, "cpu", "xavier");
    CppNet::Activations::ReLU relu5;
    CppNet::Layers::Linear layer6(30, 1, "TestLayer6", true, true, "cpu", "xavier");
    CppNet::Activations::Sigmoid sigmoid;
    CppNet::Optimizers::SGD optimizer;
    CppNet::Losses::BinaryCrossEntropy loss_fn("sum", false, 1.0);

    // Set number of threads for all layers
    layer1.set_num_threads(num_threads);
    layer2.set_num_threads(num_threads);
    layer3.set_num_threads(num_threads);
    layer4.set_num_threads(num_threads);
    layer5.set_num_threads(num_threads);
    layer6.set_num_threads(num_threads);

    // Training parameters
    double lr = 0.0004;
    int train_batch_size = 64;
    int num_train_iters = (train_data.rows() + train_batch_size - 1) / train_batch_size;

    // Indices for shuffling
    std::vector<int> train_indices(train_data.rows());
    std::iota(train_indices.begin(), train_indices.end(), 0);
    std::mt19937 g(3);

    // Start timing the training loop
    auto start_time = std::chrono::high_resolution_clock::now();

    // Training loop (reduced epochs for speed testing)
    for (int epoch = 0; epoch < epochs_to_test; epoch++) {
        double epoch_loss = 0.0;
        double epoch_acc = 0.0;

        // Shuffle training indices
        std::shuffle(train_indices.begin(), train_indices.end(), g);

        int start = 0;
        int end = std::min(train_batch_size, static_cast<int>(train_data.rows()));
        
        for (int iter = 0; iter < num_train_iters; iter++) {
            int batch_size = end - start;
            Eigen::MatrixXd x_batch_(batch_size, train_data.cols()-1);
            Eigen::MatrixXd y_batch_(batch_size, 1);

            // Slice indices
            auto first = train_indices.begin() + start;
            auto last = train_indices.begin() + end;
            std::vector<int> indices_(first, last);
            
            dp.prepare_batch(train_data, x_batch_, y_batch_, indices_);
            standardize(x_batch_);

            Eigen::TensorMap<Eigen::Tensor<double, 2>> x_batch(
                x_batch_.data(),
                x_batch_.rows(),
                x_batch_.cols()
            );

            Eigen::TensorMap<Eigen::Tensor<double, 2>> y_batch(
                y_batch_.data(), 
                y_batch_.rows(), 
                y_batch_.cols()
            );

            // Forward propagation
            Eigen::Tensor<double, 2> output1 = relu1.forward(layer1.forward(x_batch));
            Eigen::Tensor<double, 2> output2 = relu2.forward(layer2.forward(output1));
            Eigen::Tensor<double, 2> output3 = relu3.forward(layer3.forward(output2));
            Eigen::Tensor<double, 2> output4 = relu4.forward(layer4.forward(output3));
            Eigen::Tensor<double, 2> output5 = relu5.forward(layer5.forward(output4));
            Eigen::Tensor<double, 2> output6 = sigmoid.forward(layer6.forward(output5));

            // Compute loss and accuracy
            double loss = loss_fn.forward(output6, y_batch);
            epoch_loss += loss;
            Eigen::Map<Eigen::MatrixXd> output_map(output6.data(), output6.dimension(0), output6.dimension(1));
            Eigen::Map<Eigen::MatrixXd> y_map(y_batch.data(), y_batch.dimension(0), y_batch.dimension(1));

            Eigen::MatrixXd pred_matrix = (output_map.array() > 0.5).cast<double>();
            double acc = (pred_matrix.array() == y_map.array()).cast<double>().mean();
            epoch_acc += acc;

            // Reset gradients
            layer1.reset_grads();
            layer2.reset_grads();
            layer3.reset_grads();
            layer4.reset_grads();
            layer5.reset_grads();
            layer6.reset_grads();


            // Backward propagation
            Eigen::Tensor<double, 2> grad_out = loss_fn.backward(output6, y_batch);
            Eigen::Tensor<double, 2> grad_in6 = layer6.backward(sigmoid.backward(grad_out));
            Eigen::Tensor<double, 2> grad_in5 = layer5.backward(relu5.backward(grad_in6));
            Eigen::Tensor<double, 2> grad_in4 = layer4.backward(relu4.backward(grad_in5));  
            Eigen::Tensor<double, 2> grad_in3 = layer3.backward(relu3.backward(grad_in4));
            Eigen::Tensor<double, 2> grad_in2 = layer2.backward(relu2.backward(grad_in3));
            Eigen::Tensor<double, 2> grad_in1 = layer1.backward(relu1.backward(grad_in2));  
            
            // Update parameters using SGD
            // Note: Using the step function defined in the Linear layer to update weights
            layer1.step(optimizer, lr);
            layer2.step(optimizer, lr);
            layer3.step(optimizer, lr);
            layer4.step(optimizer, lr);
            layer5.step(optimizer, lr);
            layer6.step(optimizer, lr); 

            start = end;
            end = std::min(end + train_batch_size, static_cast<int>(train_data.rows()));
        }

        double mean_loss = epoch_loss / num_train_iters;
        double mean_acc = epoch_acc / num_train_iters;

        // Print progress every 10 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << " | Loss: " << std::fixed << std::setprecision(4) 
                      << mean_loss << " | Acc: " << mean_acc << std::endl;
        }
    }

    // Stop timing and calculate duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double time_seconds = duration.count() / 1000.0;
    
    std::cout << "Training completed in: " << std::fixed << std::setprecision(2) 
              << time_seconds << " seconds" << std::endl;
    
    return time_seconds;
}

int main() {
    std::cout << "=== Deep Learning Library Performance Test ===" << std::endl;
    std::cout << "Testing with different numbers of CPU threads" << std::endl;
    std::cout << "Your system has 8 CPU cores available" << std::endl;
    
    // Array of thread counts to test (you can modify this)
    std::vector<int> thread_counts = {1, 2, 4, 6, 8};
    
    // Number of epochs to run for each test (reduced for faster testing)
    int test_epochs = 1000;
    
    // Store results for comparison
    std::vector<std::pair<int, double>> results;
    
    // Test each thread count
    for (int threads : thread_counts) {
        double time_taken = run_training_with_threads(threads, test_epochs);
        results.push_back({threads, time_taken});
        
        // Small delay between tests
        std::cout << "Waiting 2 seconds before next test...\n" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    
    // Print performance comparison results
    std::cout << "\n=== PERFORMANCE COMPARISON RESULTS ===" << std::endl;
    std::cout << "Threads\t| Time (s)\t| Speedup\t| Efficiency" << std::endl;
    std::cout << "--------|---------------|---------------|----------" << std::endl;
    
    double baseline_time = results[0].second; // Single thread time as baseline
    
    for (const auto& result : results) {
        int threads = result.first;
        double time = result.second;
        double speedup = baseline_time / time;
        double efficiency = speedup / threads * 100; // Percentage efficiency
        
        std::cout << threads << "\t| " << std::fixed << std::setprecision(2) << time 
                  << "\t\t| " << speedup << "x\t\t| " << efficiency << "%" << std::endl;
    }
    
    // Find the best performing configuration
    auto best_result = *std::min_element(results.begin(), results.end(), 
                                       [](const auto& a, const auto& b) {
                                           return a.second < b.second;
                                       });
    
    std::cout << "\nBest performance: " << best_result.first << " threads with " 
              << std::fixed << std::setprecision(2) << best_result.second << " seconds" << std::endl;
    
    return 0;
}



/*
=== Deep Learning Library Performance Test ===
Testing with different numbers of CPU threads
Your system has 8 CPU cores available

=== Testing with 1 threads ===
Epoch: 0 | Loss: 41.2578 | Acc: 0.6973
Epoch: 100 | Loss: 2.5315 | Acc: 0.9874
Epoch: 200 | Loss: 1.1282 | Acc: 0.9937
Epoch: 300 | Loss: 0.7245 | Acc: 0.9961
Epoch: 400 | Loss: 0.6088 | Acc: 0.9937
Epoch: 500 | Loss: 0.1649 | Acc: 1.0000
Epoch: 600 | Loss: 0.0938 | Acc: 1.0000
Epoch: 700 | Loss: 0.0703 | Acc: 1.0000
Epoch: 800 | Loss: 0.1303 | Acc: 1.0000
Epoch: 900 | Loss: 0.0434 | Acc: 1.0000
Training completed in: 53.88 seconds
Waiting 2 seconds before next test...


=== Testing with 2 threads ===
Epoch: 0 | Loss: 43.5739 | Acc: 0.4672
Epoch: 100 | Loss: 2.2843 | Acc: 0.9893
Epoch: 200 | Loss: 1.6148 | Acc: 0.9889
Epoch: 300 | Loss: 0.7570 | Acc: 0.9980
Epoch: 400 | Loss: 0.6183 | Acc: 0.9961
Epoch: 500 | Loss: 0.3459 | Acc: 1.0000
Epoch: 600 | Loss: 0.1108 | Acc: 1.0000
Epoch: 700 | Loss: 0.0923 | Acc: 1.0000
Epoch: 800 | Loss: 0.2296 | Acc: 0.9976
Epoch: 900 | Loss: 0.1270 | Acc: 1.0000
Training completed in: 30.04 seconds
Waiting 2 seconds before next test...


=== Testing with 4 threads ===
Epoch: 0 | Loss: 42.2864 | Acc: 0.6438
Epoch: 100 | Loss: 2.1554 | Acc: 0.9874
Epoch: 200 | Loss: 1.0209 | Acc: 0.9932
Epoch: 300 | Loss: 0.5234 | Acc: 0.9961
Epoch: 400 | Loss: 0.5185 | Acc: 0.9961
Epoch: 500 | Loss: 0.1954 | Acc: 1.0000
Epoch: 600 | Loss: 0.0815 | Acc: 1.0000
Epoch: 700 | Loss: 0.0826 | Acc: 1.0000
Epoch: 800 | Loss: 0.1522 | Acc: 0.9976
Epoch: 900 | Loss: 0.0809 | Acc: 1.0000
Training completed in: 19.68 seconds
Waiting 2 seconds before next test...


=== Testing with 6 threads ===
Epoch: 0 | Loss: 42.5396 | Acc: 0.4700
Epoch: 100 | Loss: 2.2876 | Acc: 0.9893
Epoch: 200 | Loss: 0.9625 | Acc: 0.9980
Epoch: 300 | Loss: 0.5890 | Acc: 0.9980
Epoch: 400 | Loss: 0.4436 | Acc: 0.9980
Epoch: 500 | Loss: 0.1840 | Acc: 1.0000
Epoch: 600 | Loss: 0.0598 | Acc: 1.0000
Epoch: 700 | Loss: 0.0765 | Acc: 1.0000
Epoch: 800 | Loss: 0.1512 | Acc: 0.9976
Epoch: 900 | Loss: 0.1220 | Acc: 1.0000
Training completed in: 22.08 seconds
Waiting 2 seconds before next test...


=== Testing with 8 threads ===
Epoch: 0 | Loss: 42.4832 | Acc: 0.5630
Epoch: 100 | Loss: 2.2283 | Acc: 0.9874
Epoch: 200 | Loss: 1.0356 | Acc: 0.9961
Epoch: 300 | Loss: 0.5081 | Acc: 0.9980
Epoch: 400 | Loss: 0.4974 | Acc: 0.9980
Epoch: 500 | Loss: 0.1678 | Acc: 1.0000
Epoch: 600 | Loss: 0.0667 | Acc: 1.0000
Epoch: 700 | Loss: 0.0492 | Acc: 1.0000
Epoch: 800 | Loss: 0.0957 | Acc: 1.0000
Epoch: 900 | Loss: 0.0518 | Acc: 1.0000
Training completed in: 19.20 seconds
Waiting 2 seconds before next test...


=== PERFORMANCE COMPARISON RESULTS ===
Threads | Time (s)      | Speedup       | Efficiency
--------|---------------|---------------|----------
1       | 53.88         | 1.00x         | 100.00%
2       | 30.04         | 1.79x         | 89.69%
4       | 19.68         | 2.74x         | 68.44%
6       | 22.08         | 2.44x         | 40.67%
8       | 19.20         | 2.81x         | 35.09%

Best performance: 8 threads with 19.20 seconds
*/