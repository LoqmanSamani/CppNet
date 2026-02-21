/**
 * @file test_attention.cpp
 * @brief Unit tests for the MultiHeadAttention layer
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/attention.hpp"
#include "CppNet/optimizers/sgd.hpp"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                                           \
    do {                                                                     \
        std::cout << "  [TEST] " << name << " ... ";                         \
        try {

#define END_TEST(name)                                                       \
            std::cout << "PASSED" << std::endl;                              \
            ++tests_passed;                                                  \
        } catch (const std::exception& e) {                                  \
            std::cout << "FAILED: " << e.what() << std::endl;                \
            ++tests_failed;                                                  \
        } catch (...) {                                                      \
            std::cout << "FAILED (unknown exception)" << std::endl;          \
            ++tests_failed;                                                  \
        }                                                                    \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    if (std::fabs((a) - (b)) > (eps)) \
        throw std::runtime_error( \
            std::string("Expected ") + std::to_string(b) + " but got " + std::to_string(a))

#define ASSERT_TRUE(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond)

int main()
{
    std::cout << "=== MultiHeadAttention Layer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Construction - basic parameters")
    {
        CppNet::Layers::MultiHeadAttention mha(64, 4, "cpu-eigen");
        ASSERT_TRUE(mha.get_embed_dim() == 64);
        ASSERT_TRUE(mha.get_num_heads() == 4);
        ASSERT_TRUE(mha.is_trainable());
    }
    END_TEST("Construction - basic parameters");

    // ---------------------------------------------------------------
    TEST("Forward - self-attention output shape")
    {
        // Input: [batch=2, seq_len=5, embed_dim=32]
        CppNet::Layers::MultiHeadAttention mha(32, 4, "cpu-eigen");
        Eigen::Tensor<float, 3> input(2, 5, 32);
        input.setRandom();
        auto output = mha.forward(input, input, input);

        ASSERT_TRUE(output.dimension(0) == 2);
        ASSERT_TRUE(output.dimension(1) == 5);
        ASSERT_TRUE(output.dimension(2) == 32);
    }
    END_TEST("Forward - self-attention output shape");

    // ---------------------------------------------------------------
    TEST("Forward - output is finite")
    {
        CppNet::Layers::MultiHeadAttention mha(16, 2, "cpu-eigen");
        Eigen::Tensor<float, 3> input(1, 4, 16);
        input.setRandom();
        auto output = mha.forward(input, input, input);

        for (int i = 0; i < output.size(); ++i)
            ASSERT_TRUE(std::isfinite(output.data()[i]));
    }
    END_TEST("Forward - output is finite");

    // ---------------------------------------------------------------
    TEST("Forward - different Q, K, V (cross-attention shape)")
    {
        CppNet::Layers::MultiHeadAttention mha(32, 4, "cpu-eigen");
        Eigen::Tensor<float, 3> query(1, 3, 32);
        Eigen::Tensor<float, 3> key(1, 5, 32);
        Eigen::Tensor<float, 3> value(1, 5, 32);
        query.setRandom();
        key.setRandom();
        value.setRandom();

        auto output = mha.forward(query, key, value);

        // output shape follows query's seq_len
        ASSERT_TRUE(output.dimension(0) == 1);
        ASSERT_TRUE(output.dimension(1) == 3);
        ASSERT_TRUE(output.dimension(2) == 32);
    }
    END_TEST("Forward - different Q, K, V (cross-attention shape)");

    // ---------------------------------------------------------------
    TEST("Backward - grad shape matches input")
    {
        CppNet::Layers::MultiHeadAttention mha(32, 4, "cpu-eigen");
        Eigen::Tensor<float, 3> input(1, 4, 32);
        input.setRandom();
        auto output = mha.forward(input, input, input);

        Eigen::Tensor<float, 3> grad_output(output.dimensions());
        grad_output.setConstant(0.01f);
        auto grad_input = mha.backward(grad_output);

        ASSERT_TRUE(grad_input.dimension(0) == 1);
        ASSERT_TRUE(grad_input.dimension(1) == 4);
        ASSERT_TRUE(grad_input.dimension(2) == 32);
    }
    END_TEST("Backward - grad shape matches input");

    // ---------------------------------------------------------------
    TEST("Backward - grad is finite")
    {
        CppNet::Layers::MultiHeadAttention mha(16, 2, "cpu-eigen");
        Eigen::Tensor<float, 3> input(1, 3, 16);
        input.setRandom();
        auto output = mha.forward(input, input, input);

        Eigen::Tensor<float, 3> grad_output(output.dimensions());
        grad_output.setConstant(0.01f);
        auto grad_input = mha.backward(grad_output);

        for (int i = 0; i < grad_input.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad_input.data()[i]));
    }
    END_TEST("Backward - grad is finite");

    // ---------------------------------------------------------------
    TEST("Freeze/Unfreeze")
    {
        CppNet::Layers::MultiHeadAttention mha(32, 4, "cpu-eigen");
        ASSERT_TRUE(mha.is_trainable());
        mha.freeze();
        ASSERT_TRUE(!mha.is_trainable());
        mha.unfreeze();
        ASSERT_TRUE(mha.is_trainable());
    }
    END_TEST("Freeze/Unfreeze");

    // ---------------------------------------------------------------
    TEST("Single head attention")
    {
        CppNet::Layers::MultiHeadAttention mha(16, 1, "cpu-eigen");
        Eigen::Tensor<float, 3> input(1, 3, 16);
        input.setRandom();
        auto output = mha.forward(input, input, input);

        ASSERT_TRUE(output.dimension(0) == 1);
        ASSERT_TRUE(output.dimension(1) == 3);
        ASSERT_TRUE(output.dimension(2) == 16);
    }
    END_TEST("Single head attention");

    // ---------------------------------------------------------------
    TEST("Single timestep sequence")
    {
        CppNet::Layers::MultiHeadAttention mha(32, 4, "cpu-eigen");
        Eigen::Tensor<float, 3> input(1, 1, 32);
        input.setRandom();
        auto output = mha.forward(input, input, input);

        ASSERT_TRUE(output.dimension(1) == 1);
    }
    END_TEST("Single timestep sequence");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
