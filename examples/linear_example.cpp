#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>





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




int main() 
{
    // load and split breast cancer dataset
    DataProcessor dp(true);
    Eigen::MatrixXd data = dp.load_data("../examples/breast_cancer.csv", 31, 1.0);
    Eigen::MatrixXd train_data(500, 31);
    Eigen::MatrixXd test_data(39, 31);
    Eigen::MatrixXd val_data(30, 31);
    dp.split_data(data, train_data, test_data, val_data, true);

    // model with two hidden layers
    CppNet::Linear in_layer(30, 50, "TestLayer1", true, true);
    CppNet::ReLU relu1;
    CppNet::Linear hid1(50, 50, "TestLayer2", true, true);
    CppNet::ReLU relu2;
    CppNet::Linear hid2(50, 30, "TestLayer3", true, true);
    CppNet::ReLU relu3;
    CppNet::Linear out_layer(30, 1, "TestLayer4", true, true);
    CppNet::Sigmoid sigmoid;
    CppNet::SGD optimizer;
    CppNet::BinaryCrossEntropy loss_fn;

    // training parameters
    int epochs = 1000;
    double lr = 0.0004;
    int train_batch_size = 64;
    int val_test_batch_size = 10;
    int num_train_iters = (train_data.rows() + train_batch_size - 1) / train_batch_size;
    int num_val_iters = (val_data.rows() + val_test_batch_size - 1) / val_test_batch_size;
    int num_test_iters = (test_data.rows() + val_test_batch_size - 1) / val_test_batch_size;

    // indices for shuffling
    std::vector<int> train_indices(train_data.rows());
    std::iota(train_indices.begin(), train_indices.end(), 0);
    std::mt19937 g(3);
    
    std::vector<int> val_indices(val_data.rows());
    std::iota(val_indices.begin(), val_indices.end(), 0);
    std::mt19937 r(3);
    
    std::vector<int> test_indices(test_data.rows());
    std::iota(test_indices.begin(), test_indices.end(), 0);
    std::mt19937 t(3);

    // training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        double epoch_acc = 0.0;

        // shuffle training indices
        std::shuffle(train_indices.begin(), train_indices.end(), g);

        int start = 0;
        int end = std::min(train_batch_size, static_cast<int>(train_data.rows()));
        
        for (int iter = 0; iter < num_train_iters; iter++) {
            int batch_size = end - start;
            Eigen::MatrixXd x_batch(batch_size, train_data.cols()-1);
            Eigen::MatrixXd y_batch(batch_size, 1);

            // slice indices
            auto first = train_indices.begin() + start;
            auto last = train_indices.begin() + end;
            std::vector<int> indices_(first, last);
            
            dp.prepare_batch(train_data, x_batch, y_batch, indices_);
            standardize(x_batch);

            // forward propagation
            Eigen::MatrixXd output1 = relu1.forward(in_layer.forward(x_batch));
            Eigen::MatrixXd output2 = relu2.forward(hid1.forward(output1));
            Eigen::MatrixXd output3 = relu3.forward(hid2.forward(output2));
            Eigen::MatrixXd output4 = sigmoid.forward(out_layer.forward(output3));

            // compute loss and accuracy
            double loss = loss_fn.forward(y_batch, output4);
            epoch_loss += loss;
            Eigen::MatrixXd pred = (output4.array() > 0.5).cast<double>();
            double acc = (pred.array() == y_batch.array()).cast<double>().mean();
            epoch_acc += acc;

            // reset gradients
            in_layer.reset_grads();
            hid1.reset_grads();
            hid2.reset_grads();
            out_layer.reset_grads();

            // backward propagation
            Eigen::MatrixXd grad_out = loss_fn.backward(output4, y_batch);
            Eigen::MatrixXd grad_in1 = out_layer.backward(sigmoid.backward(grad_out));
            Eigen::MatrixXd grad_in2 = hid2.backward(relu3.backward(grad_in1));
            Eigen::MatrixXd grad_in3 = hid1.backward(relu2.backward(grad_in2));
            Eigen::MatrixXd grad_in4 = in_layer.backward(relu1.backward(grad_in3));

            // update parameters
            in_layer.update_parameters(optimizer, lr);
            hid1.update_parameters(optimizer, lr);
            hid2.update_parameters(optimizer, lr);
            out_layer.update_parameters(optimizer, lr);

            start = end;
            end = std::min(end + train_batch_size, static_cast<int>(train_data.rows()));
        }

        double mean_loss = epoch_loss / num_train_iters;
        double mean_acc = epoch_acc / num_train_iters;

        // validation
        double val_loss = 0.0;
        double val_acc = 0.0;
        if (epoch % 200 == 0) {
            std::shuffle(val_indices.begin(), val_indices.end(), r);
            int start = 0;
            int end = std::min(val_test_batch_size, static_cast<int>(val_data.rows()));
            
            for (int iter = 0; iter < num_val_iters; iter++) {
                int batch_size = end - start;
                Eigen::MatrixXd x_batch(batch_size, val_data.cols()-1);
                Eigen::MatrixXd y_batch(batch_size, 1);
            
                auto first = val_indices.begin() + start;
                auto last = val_indices.begin() + end;
                std::vector<int> indices_(first, last);
                
                dp.prepare_batch(val_data, x_batch, y_batch, indices_);
                standardize(x_batch);

                // forward propagation
                Eigen::MatrixXd output1 = relu1.forward(in_layer.forward(x_batch));
                Eigen::MatrixXd output2 = relu2.forward(hid1.forward(output1));
                Eigen::MatrixXd output3 = relu3.forward(hid2.forward(output2));
                Eigen::MatrixXd output4 = sigmoid.forward(out_layer.forward(output3));

                // compute loss and accuracy
                val_loss += loss_fn.forward(y_batch, output4);
                Eigen::MatrixXd pred = (output4.array() > 0.5).cast<double>();
                val_acc += (pred.array() == y_batch.array()).cast<double>().mean();

                start = end;
                end = std::min(end + val_test_batch_size, static_cast<int>(val_data.rows()));
            }
            val_loss /= num_val_iters;
            val_acc /= num_val_iters;
        }

        // print progress
        if (epoch % 50 == 0) {
            std::cout << "Epoch: " << epoch << " | Loss: " << mean_loss << " | Acc: " << mean_acc << std::endl;
            if (epoch % 200 == 0) {
                std::cout << "Val Loss: " << val_loss << " | Val Acc: " << val_acc << std::endl;
            }
        }
    }

    // test phase
    double test_loss = 0.0;
    double test_acc = 0.0;
    std::shuffle(test_indices.begin(), test_indices.end(), t);
    int start = 0;
    int end = std::min(val_test_batch_size, static_cast<int>(test_data.rows()));
    
    for (int iter = 0; iter < num_test_iters; iter++) {
        int batch_size = end - start;
        Eigen::MatrixXd x_batch(batch_size, test_data.cols()-1);
        Eigen::MatrixXd y_batch(batch_size, 1);
    
        auto first = test_indices.begin() + start;
        auto last = test_indices.begin() + end;
        std::vector<int> indices_(first, last);
        
        dp.prepare_batch(test_data, x_batch, y_batch, indices_);
        standardize(x_batch);

        // forward propagation
        Eigen::MatrixXd output1 = relu1.forward(in_layer.forward(x_batch));
        Eigen::MatrixXd output2 = relu2.forward(hid1.forward(output1));
        Eigen::MatrixXd output3 = relu3.forward(hid2.forward(output2));
        Eigen::MatrixXd output4 = sigmoid.forward(out_layer.forward(output3));

        // compute loss and accuracy
        test_loss += loss_fn.forward(y_batch, output4);
        Eigen::MatrixXd pred = (output4.array() > 0.5).cast<double>();
        test_acc += (pred.array() == y_batch.array()).cast<double>().mean();

        start = end;
        end = std::min(end + val_test_batch_size, static_cast<int>(test_data.rows()));
    }
    test_loss /= num_test_iters;
    test_acc /= num_test_iters;

    std::cout << "Test Loss: " << test_loss << " | Test Acc: " << test_acc << std::endl;

    return 0;
}

/*
Epoch: 0 | Loss: 0.713147 | Acc: 0.535156
Val Loss: 0.683769 | Val Acc: 0.6
Epoch: 50 | Loss: 0.630926 | Acc: 0.70598
Epoch: 100 | Loss: 0.559367 | Acc: 0.835487
Epoch: 150 | Loss: 0.498512 | Acc: 0.883113
Epoch: 200 | Loss: 0.448419 | Acc: 0.894832
Val Loss: 0.426632 | Val Acc: 0.966667
Epoch: 250 | Loss: 0.403878 | Acc: 0.905499
Epoch: 300 | Loss: 0.364154 | Acc: 0.919171
Epoch: 350 | Loss: 0.327517 | Acc: 0.928786
Epoch: 400 | Loss: 0.298371 | Acc: 0.937049
Val Loss: 0.335642 | Val Acc: 0.966667
Epoch: 450 | Loss: 0.275099 | Acc: 0.936148
Epoch: 500 | Loss: 0.252413 | Acc: 0.934195
Epoch: 550 | Loss: 0.236108 | Acc: 0.936599
Epoch: 600 | Loss: 0.218971 | Acc: 0.939153
Val Loss: 0.295053 | Val Acc: 0.866667
Epoch: 650 | Loss: 0.211909 | Acc: 0.943059
Epoch: 700 | Loss: 0.189925 | Acc: 0.944862
Epoch: 750 | Loss: 0.19102 | Acc: 0.945463
Epoch: 800 | Loss: 0.179737 | Acc: 0.941106
Val Loss: 0.376934 | Val Acc: 0.833333
Epoch: 850 | Loss: 0.169806 | Acc: 0.945463
Epoch: 900 | Loss: 0.166353 | Acc: 0.94351
Epoch: 950 | Loss: 0.168052 | Acc: 0.938101
Test Loss: 0.187232 | Test Acc: 0.947222
*/