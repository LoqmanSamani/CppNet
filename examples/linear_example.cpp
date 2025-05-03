#include "layers.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
    CppNet::Linear layer(3, 2); // 3 input features, 2 output units
    Eigen::MatrixXd X(2, 3); // Batch of 2 samples, 3 features
    X << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Eigen::MatrixXd output = layer.forward(X);
    std::cout << "Output:\n" << output << std::endl;
    Eigen::MatrixXd grad_output(2, 2); // Gradient from next layer
    grad_output.setOnes();
    Eigen::MatrixXd grad_input = layer.backward(grad_output);
    layer.update_params(0.01); // Learning rate = 0.01
    return 0;
}