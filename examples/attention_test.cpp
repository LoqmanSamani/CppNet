#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <vector>
#include <iomanip>
#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h>

#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <random>



int main() 
{
   
    CppNet::Layers::MultiHeadAttention mha(256, 256, 8, 512, 0.1, true, false, "MHA_Test", "cpu", "xavier", 10000);
    mha.print_layer_info();

    Eigen::Tensor<double, 3> inputs(2, 10, 256); // batch_size=2, seq_length=10, in_size=256
    inputs.setRandom();
    Eigen::Tensor<double, 3> targets(2, 10, 256); // batch_size=2, seq_length=10, in_size=256
    targets.setRandom();

    Eigen::Tensor<double, 3> outputs = mha.forward(inputs, targets, true);
    std::cout << "Output shape: [" << outputs.dimension(0) << ", " << outputs.dimension(1) << ", " << outputs.dimension(2) << "]" << std::endl;
    Eigen::Tensor<double, 3> grad_outputs(2, 10, 256); // Same shape as outputs
    grad_outputs.setRandom();
    Eigen::Tensor<double, 3> grad_targets;
    Eigen::Tensor<double, 3> grad_inputs = mha.backward(grad_outputs, grad_targets);
    std::cout << "Grad Inputs shape: [" << grad_inputs.dimension(0) << ", " << grad_inputs.dimension(1) << ", " << grad_inputs.dimension(2) << "]" << std::endl;
    std::cout << "Grad Targets shape: [" << grad_targets.dimension(0) << ", " << grad_targets.dimension(1) << ", " << grad_targets.dimension(2) << "]" << std::endl;


    return 0;

}
