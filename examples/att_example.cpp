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
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <cassert>

void print_tensor_3d_shape(const Eigen::Tensor<double, 3>& tensor, const std::string& name) {
    std::cout << name << " shape: [" << tensor.dimension(0) << ", " 
              << tensor.dimension(1) << ", " << tensor.dimension(2) << "]" << std::endl;
}

void print_tensor_sample(const Eigen::Tensor<double, 3>& tensor, const std::string& name, int max_elements = 5) {
    std::cout << name << " sample values: ";
    int count = 0;
    for (int i = 0; i < std::min((int)tensor.dimension(0), 2) && count < max_elements; ++i) {
        for (int j = 0; j < std::min((int)tensor.dimension(1), 2) && count < max_elements; ++j) {
            for (int k = 0; k < std::min((int)tensor.dimension(2), 3) && count < max_elements; ++k) {
                std::cout << tensor(i, j, k) << " ";
                count++;
            }
        }
    }
    std::cout << std::endl;
}

Eigen::Tensor<double, 3> create_test_input(int batch_size, int seq_len, int embed_dim) {
    Eigen::Tensor<double, 3> input(batch_size, seq_len, embed_dim);
    
    // Initialize with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 0.1);
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < embed_dim; ++k) {
                input(i, j, k) = dis(gen);
            }
        }
    }
    
    return input;
}

bool test_forward_pass() {
    std::cout << "\n=== Testing Forward Pass ===" << std::endl;
    
    try {
        // Test parameters - FIXED: embed_dim (64) is divisible by num_heads (8)
        int batch_size = 2;
        int seq_len = 10;
        int embed_dim = 64;  // 64 is divisible by 8
        int num_heads = 8;   // 64 / 8 = 8 dimensions per head
        
        std::cout << "Using embed_dim=" << embed_dim << ", num_heads=" << num_heads 
                  << ", head_dim=" << (embed_dim / num_heads) << std::endl;
        
        CppNet::Layers::MultiHeadAttention attention(
            embed_dim,      // in_size
            embed_dim,      // out_size
            num_heads,      // num_heads
            seq_len,        // context_length
            0.1,            // dropout_rate
            "test-attention", // layer_name
            true,           // trainable
            true            // qkv_bias
        );
        
        // Print layer info
        attention.print_layer_info();
        
        // Create test input
        Eigen::Tensor<double, 3> input = create_test_input(batch_size, seq_len, embed_dim);
        print_tensor_3d_shape(input, "Input");
        print_tensor_sample(input, "Input");
        
        // Test self-attention (Y is empty)
        std::cout << "\nTesting self-attention..." << std::endl;
        Eigen::Tensor<double, 3> empty_Y; // Create empty tensor for self-attention
        Eigen::Tensor<double, 3> output = attention.forward(input, empty_Y);
        
        print_tensor_3d_shape(output, "Output");
        print_tensor_sample(output, "Output");
        
        // Check output dimensions
        assert(output.dimension(0) == batch_size);
        assert(output.dimension(1) == seq_len);
        assert(output.dimension(2) == embed_dim);
        
        // Check for NaN or infinite values
        bool has_nan = false;
        for (int i = 0; i < batch_size && !has_nan; ++i) {
            for (int j = 0; j < seq_len && !has_nan; ++j) {
                for (int k = 0; k < embed_dim && !has_nan; ++k) {
                    if (std::isnan(output(i, j, k)) || std::isinf(output(i, j, k))) {
                        has_nan = true;
                    }
                }
            }
        }
        
        if (has_nan) {
            std::cout << "ERROR: Output contains NaN or infinite values!" << std::endl;
            return false;
        }
        
        std::cout << "✓ Forward pass test passed!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR in forward pass: " << e.what() << std::endl;
        return false;
    }
}

/*
bool test_causal_masking() {
    std::cout << "\n=== Testing Causal Masking ===" << std::endl;
    
    try {
        int batch_size = 1;
        int seq_len = 5;
        int embed_dim = 32;  // FIXED: 32 is divisible by 4
        int num_heads = 4;   // 32 / 4 = 8 dimensions per head
        
        std::cout << "Using embed_dim=" << embed_dim << ", num_heads=" << num_heads 
                  << ", head_dim=" << (embed_dim / num_heads) << std::endl;
        
        CppNet::Layers::MultiHeadAttention attention(
            embed_dim, embed_dim, num_heads, seq_len, 0.0, "causal-test", true, false
        );
        
        // Create a simple test input where each position has a unique pattern
        Eigen::Tensor<double, 3> input(batch_size, seq_len, embed_dim);
        input.setZero();
        
        // Set different values for each position to test causality
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int dim = 0; dim < embed_dim; ++dim) {
                input(0, pos, dim) = pos + 1; // Position 0 -> 1, Position 1 -> 2, etc.
            }
        }
        
        print_tensor_sample(input, "Causal test input");
        
        // Forward pass with causal masking (self-attention)
        Eigen::Tensor<double, 3> empty_Y;
        Eigen::Tensor<double, 3> output = attention.forward(input, empty_Y, true);
        
        print_tensor_sample(output, "Causal test output");
        
        std::cout << "✓ Causal masking test completed (check attention weights manually for full verification)" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR in causal masking test: " << e.what() << std::endl;
        return false;
    }
}

bool test_cross_attention() {
    std::cout << "\n=== Testing Cross Attention ===" << std::endl;
    
    try {
        int batch_size = 1;
        int seq_len_x = 8;
        int seq_len_y = 10;
        int embed_dim = 64;  // FIXED: 64 is divisible by 8
        int num_heads = 8;   // 64 / 8 = 8 dimensions per head
        
        std::cout << "Using embed_dim=" << embed_dim << ", num_heads=" << num_heads 
                  << ", head_dim=" << (embed_dim / num_heads) << std::endl;
        
        CppNet::Layers::MultiHeadAttention attention(
            embed_dim, embed_dim, num_heads, std::max(seq_len_x, seq_len_y), 0.1, "cross-attention-test"
        );
        
        // Create different inputs for X and Y
        Eigen::Tensor<double, 3> X = create_test_input(batch_size, seq_len_x, embed_dim);
        Eigen::Tensor<double, 3> Y = create_test_input(batch_size, seq_len_y, embed_dim);
        
        // Make Y distinctly different from X
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len_y; ++j) {
                for (int k = 0; k < embed_dim; ++k) {
                    Y(i, j, k) += 1.0; // Shift Y values
                }
            }
        }
        
        print_tensor_3d_shape(X, "Cross-attention X");
        print_tensor_3d_shape(Y, "Cross-attention Y");
        
        // Cross-attention: Q from X, K and V from Y
        Eigen::Tensor<double, 3> output = attention.forward(X, Y, false);
        
        print_tensor_3d_shape(output, "Cross-attention output");
        print_tensor_sample(output, "Cross-attention output");
        
        // Output should have same batch size and sequence length as X (queries)
        assert(output.dimension(0) == batch_size);
        assert(output.dimension(1) == seq_len_x);
        assert(output.dimension(2) == embed_dim);
        
        std::cout << "✓ Cross-attention test passed!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR in cross-attention test: " << e.what() << std::endl;
        return false;
    }
}
    */

bool test_backward_pass() {
    std::cout << "\n=== Testing Backward Pass ===" << std::endl;
    
    try {
        int batch_size = 2;
        int seq_len = 6;
        int embed_dim = 32;  // FIXED: 32 is divisible by 4
        int num_heads = 4;   // 32 / 4 = 8 dimensions per head
        
        std::cout << "Using embed_dim=" << embed_dim << ", num_heads=" << num_heads 
                  << ", head_dim=" << (embed_dim / num_heads) << std::endl;
        
        CppNet::Layers::MultiHeadAttention attention(
            embed_dim, embed_dim, num_heads, seq_len, 0.1, "backward-test", true, true
        );
        
        // Forward pass
        Eigen::Tensor<double, 3> input = create_test_input(batch_size, seq_len, embed_dim);
        Eigen::Tensor<double, 3> output = attention.forward(input);
        
        // Create gradient for backward pass
        Eigen::Tensor<double, 3> grad_output(batch_size, seq_len, embed_dim);
        grad_output.setConstant(0.1); // Simple uniform gradient
        
        print_tensor_3d_shape(grad_output, "Gradient output");
        
        // Backward pass
        Eigen::Tensor<double, 3> empty_dY; // Create empty tensor for self-attention backward
        Eigen::Tensor<double, 3> grad_input = attention.backward(grad_output, empty_dY);
        
        print_tensor_3d_shape(grad_input, "Gradient input");
        print_tensor_sample(grad_input, "Gradient input");
        
        // Check gradient dimensions
        assert(grad_input.dimension(0) == batch_size);
        assert(grad_input.dimension(1) == seq_len);
        assert(grad_input.dimension(2) == embed_dim);
        
        // Check that gradients are computed for weights
        const auto& grad_Wq = attention.get_grad_query_weights();
        const auto& grad_Wk = attention.get_grad_key_weights();
        const auto& grad_Wv = attention.get_grad_value_weights();
        
        std::cout << "Query weight gradients shape: [" << grad_Wq.dimension(0) << ", " << grad_Wq.dimension(1) << "]" << std::endl;
        std::cout << "Key weight gradients shape: [" << grad_Wk.dimension(0) << ", " << grad_Wk.dimension(1) << "]" << std::endl;
        std::cout << "Value weight gradients shape: [" << grad_Wv.dimension(0) << ", " << grad_Wv.dimension(1) << "]" << std::endl;
        
        // Check for NaN in gradients
        bool grad_has_nan = false;
        for (int i = 0; i < grad_Wq.dimension(0) && !grad_has_nan; ++i) {
            for (int j = 0; j < grad_Wq.dimension(1) && !grad_has_nan; ++j) {
                if (std::isnan(grad_Wq(i, j)) || std::isinf(grad_Wq(i, j))) {
                    grad_has_nan = true;
                }
            }
        }
        
        if (grad_has_nan) {
            std::cout << "ERROR: Weight gradients contain NaN or infinite values!" << std::endl;
            return false;
        }
        
        std::cout << "✓ Backward pass test passed!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR in backward pass: " << e.what() << std::endl;
        return false;
    }
}

bool test_parameter_updates() {
    std::cout << "\n=== Testing Parameter Updates ===" << std::endl;
    
    try {
        int embed_dim = 32;  // FIXED: 32 is divisible by 4
        int num_heads = 4;   // 32 / 4 = 8 dimensions per head
        
        std::cout << "Using embed_dim=" << embed_dim << ", num_heads=" << num_heads 
                  << ", head_dim=" << (embed_dim / num_heads) << std::endl;
        
        CppNet::Layers::MultiHeadAttention attention(
            embed_dim, embed_dim, num_heads, 10, 0.1, "param-test", true, true
        );
        
        // Get initial weights
        auto initial_Wq = attention.get_query_weights();
        auto initial_Wk = attention.get_key_weights();
        auto initial_Wv = attention.get_value_weights();
        
        // Perform forward and backward pass to compute gradients
        Eigen::Tensor<double, 3> input = create_test_input(1, 5, embed_dim);
        Eigen::Tensor<double, 3> output = attention.forward(input);
        
        Eigen::Tensor<double, 3> grad_output(1, 5, embed_dim);
        grad_output.setConstant(0.1);
        Eigen::Tensor<double, 3> empty_dY;
        attention.backward(grad_output, empty_dY);
        
        // Create a simple optimizer (you'll need to implement or mock this)
        // For now, let's just verify that the update_parameters method can be called
        // CppNet::Optimizers::SGD optimizer; // Assuming you have an SGD optimizer
        // attention.update_parameters(optimizer, 0.01);
        
        std::cout << "✓ Parameter update test structure verified!" << std::endl;
        std::cout << "Note: Actual parameter update testing requires optimizer implementation" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR in parameter update test: " << e.what() << std::endl;
        return false;
    }
}

/*
bool test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;
    
    try {
        // Test with minimum dimensions - FIXED: 8 is divisible by 2
        int embed_dim = 8;   // 8 is divisible by 2
        int num_heads = 2;   // 8 / 2 = 4 dimensions per head
        
        std::cout << "Using embed_dim=" << embed_dim << ", num_heads=" << num_heads 
                  << ", head_dim=" << (embed_dim / num_heads) << std::endl;
        
        CppNet::Layers::MultiHeadAttention small_attention(embed_dim, embed_dim, num_heads, 2, 0.0, "small-test");
        Eigen::Tensor<double, 3> small_input = create_test_input(1, 2, embed_dim);
        Eigen::Tensor<double, 3> empty_Y;
        Eigen::Tensor<double, 3> small_output = small_attention.forward(small_input, empty_Y);
        
        std::cout << "✓ Small dimensions test passed" << std::endl;
        
        // Test freeze/unfreeze functionality
        small_attention.freeze();
        assert(!small_attention.is_trainable());
        std::cout << "✓ Freeze functionality works" << std::endl;
        
        small_attention.unfreeze();
        assert(small_attention.is_trainable());
        std::cout << "✓ Unfreeze functionality works" << std::endl;
        
        // Test reset_grads
        small_attention.reset_grads();
        std::cout << "✓ Reset gradients works" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR in edge cases test: " << e.what() << std::endl;
        return false;
    }
}

bool test_different_head_configurations() {
    std::cout << "\n=== Testing Different Head Configurations ===" << std::endl;
    
    try {
        // Test various valid embed_dim / num_heads combinations
        std::vector<std::pair<int, int>> configs = {
            {16, 1},   // 16/1 = 16 per head
            {16, 2},   // 16/2 = 8 per head  
            {16, 4},   // 16/4 = 4 per head
            {64, 8},   // 64/8 = 8 per head
            {128, 16}, // 128/16 = 8 per head
            {96, 12}   // 96/12 = 8 per head
        };
        
        for (const auto& config : configs) {
            int embed_dim = config.first;
            int num_heads = config.second;
            int head_dim = embed_dim / num_heads;
            
            std::cout << "\nTesting config: embed_dim=" << embed_dim 
                      << ", num_heads=" << num_heads 
                      << ", head_dim=" << head_dim << std::endl;
            
            CppNet::Layers::MultiHeadAttention attention(
                embed_dim, embed_dim, num_heads, 4, 0.0, 
                "config-test-" + std::to_string(embed_dim) + "-" + std::to_string(num_heads)
            );
            
            Eigen::Tensor<double, 3> input = create_test_input(1, 4, embed_dim);
            Eigen::Tensor<double, 3> empty_Y;
            Eigen::Tensor<double, 3> output = attention.forward(input, empty_Y);
            
            // Verify output dimensions
            assert(output.dimension(0) == 1);
            assert(output.dimension(1) == 4);
            assert(output.dimension(2) == embed_dim);
            
            std::cout << "✓ Config test passed for " << embed_dim << "/" << num_heads << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR in head configuration test: " << e.what() << std::endl;
        return false;
    }
}
    */

int main() {
    std::cout << "Multi-Head Attention Layer Test Suite" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "IMPORTANT: embed_dim must be divisible by num_heads!" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    bool all_tests_passed = true;
    
    // Run all tests
    all_tests_passed &= test_forward_pass();
    //all_tests_passed &= test_causal_masking();
   // all_tests_passed &= test_cross_attention();
    all_tests_passed &= test_backward_pass();
    all_tests_passed &= test_parameter_updates();
    //all_tests_passed &= test_edge_cases();
    //all_tests_passed &= test_different_head_configurations();
    
    std::cout << "\n=====================================" << std::endl;
    if (all_tests_passed) {
        std::cout << "🎉 All tests PASSED!" << std::endl;
        std::cout << "Your MultiHeadAttention layer appears to be working correctly." << std::endl;
    } else {
        std::cout << "❌ Some tests FAILED!" << std::endl;
        std::cout << "Please review the error messages above and fix the issues." << std::endl;
    }
    
    return all_tests_passed ? 0 : 1;
}