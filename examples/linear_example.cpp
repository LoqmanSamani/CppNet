#include "layers.hpp"
#include "optimizers.hpp"
#include <Eigen/Dense>
#include <iostream>





int main() {

    CppNet::Linear layer(3, 2, "TestLayer", true, true);
    CppNet::SGD optimizer;
    Eigen::MatrixXd X(2, 3);
    X << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    Eigen::MatrixXd output = layer.forward(X);
    std::cout << "Output:\n" << output << std::endl;
    Eigen::MatrixXd grad_out(2, 2);
    grad_out.setOnes();
    layer.reset_gradients(); // Reset gradients before backward
    Eigen::MatrixXd grad_in = layer.backward(grad_out);
    layer.update_parameters(optimizer, 0.01); // learning_rate = 0.01
    std::cout << "Updated Weights:\n" << layer.get_weights() << std::endl;
    
    return 0;
}