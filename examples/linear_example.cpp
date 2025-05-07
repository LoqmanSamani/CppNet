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
    Eigen::MatrixXd train_data(450, 31);
    Eigen::MatrixXd test_data(69, 31);
    Eigen::MatrixXd val_data(50, 31);
    dp.split_data(data, train_data, test_data, val_data, true);

    // model with two hidden layers
    CppNet::Linear in_layer(30, 40, "TestLayer1", true, true);
    CppNet::ReLU relu1;
    CppNet::Linear hid1(40, 50, "TestLayer2", true, true);
    CppNet::ReLU relu2;
    CppNet::Linear hid2(50, 20, "TestLayer3", true, true);
    CppNet::ReLU relu3;
    CppNet::Linear out_layer(20, 1, "TestLayer4", true, true);
    CppNet::Sigmoid sigmoid;
    CppNet::SGD optimizer;
    CppNet::BinaryCrossEntropy loss_fn;

    // training parameters
    int epochs = 500;
    double lr = 0.001;
    int train_batch_size = 50;
    int val_test_batch_size = 10;
    int num_train_iters = (train_data.rows() + train_batch_size - 1) / train_batch_size;
    int num_val_iters = (val_data.rows() + val_test_batch_size - 1) / val_test_batch_size;
    int num_test_iters = (test_data.rows() + val_test_batch_size - 1) / val_test_batch_size;

    // indices for shuffling
    std::vector<int> train_indices(train_data.rows());
    std::iota(train_indices.begin(), train_indices.end(), 0);
    std::mt19937 g(42);
    
    std::vector<int> val_indices(val_data.rows());
    std::iota(val_indices.begin(), val_indices.end(), 0);
    std::mt19937 r(42);
    
    std::vector<int> test_indices(test_data.rows());
    std::iota(test_indices.begin(), test_indices.end(), 0);
    std::mt19937 t(42);

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
        if (epoch % 100 == 0) {
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
        if (epoch % 20 == 0) {
            std::cout << "Epoch: " << epoch << " | Loss: " << mean_loss << " | Acc: " << mean_acc << std::endl;
            if (epoch % 100 == 0) {
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
Epoch: 0 | Loss: 0.67428 | Acc: 0.604444
Val Loss: 0.656049 | Val Acc: 0.64
Epoch: 20 | Loss: 0.592645 | Acc: 0.815556
Epoch: 40 | Loss: 0.522155 | Acc: 0.877778
Epoch: 60 | Loss: 0.455956 | Acc: 0.877778
Epoch: 80 | Loss: 0.405547 | Acc: 0.893333
Epoch: 100 | Loss: 0.346777 | Acc: 0.92
Val Loss: 0.34577 | Val Acc: 0.96
Epoch: 120 | Loss: 0.306703 | Acc: 0.924444
Epoch: 140 | Loss: 0.275761 | Acc: 0.922222
Epoch: 160 | Loss: 0.244231 | Acc: 0.937778
Epoch: 180 | Loss: 0.231225 | Acc: 0.928889
Epoch: 200 | Loss: 0.21449 | Acc: 0.935556
Val Loss: 0.237781 | Val Acc: 0.94
Epoch: 220 | Loss: 0.181874 | Acc: 0.955556
Epoch: 240 | Loss: 0.181832 | Acc: 0.942222
Epoch: 260 | Loss: 0.177714 | Acc: 0.94
Epoch: 280 | Loss: 0.148196 | Acc: 0.957778
Epoch: 300 | Loss: 0.148574 | Acc: 0.946667
Val Loss: 0.181826 | Val Acc: 0.98
Epoch: 320 | Loss: 0.133869 | Acc: 0.96
Epoch: 340 | Loss: 0.129909 | Acc: 0.962222
Epoch: 360 | Loss: 0.131484 | Acc: 0.96
Epoch: 380 | Loss: 0.113087 | Acc: 0.975556
Epoch: 400 | Loss: 0.13055 | Acc: 0.948889
Val Loss: 0.224183 | Val Acc: 0.92
Epoch: 420 | Loss: 0.121789 | Acc: 0.96
Epoch: 440 | Loss: 0.112659 | Acc: 0.962222
Epoch: 460 | Loss: 0.133038 | Acc: 0.953333
Epoch: 480 | Loss: 0.0977687 | Acc: 0.977778
Epoch: 500 | Loss: 0.0930211 | Acc: 0.973333
Val Loss: 0.50318 | Val Acc: 0.84
Epoch: 520 | Loss: 0.109315 | Acc: 0.966667
Epoch: 540 | Loss: 0.103423 | Acc: 0.964444
Epoch: 560 | Loss: 0.100491 | Acc: 0.973333
Epoch: 580 | Loss: 0.104932 | Acc: 0.971111
Epoch: 600 | Loss: 0.0815216 | Acc: 0.98
Val Loss: 0.208696 | Val Acc: 0.96
Epoch: 620 | Loss: 0.0874877 | Acc: 0.971111
Epoch: 640 | Loss: 0.109313 | Acc: 0.96
Epoch: 660 | Loss: 0.0825294 | Acc: 0.977778
Epoch: 680 | Loss: 0.0802216 | Acc: 0.975556
Test Loss: 0.3449 | Test Acc: 0.9
*/