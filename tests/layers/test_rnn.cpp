/**
 * @file test_rnn.cpp
 * @brief Unit tests for the RNN layer
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/rnn.hpp"
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
    std::cout << "=== RNN Layer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Construction - basic parameters")
    {
        CppNet::Layers::RNN rnn(10, 20, true, "cpu-eigen");
        ASSERT_TRUE(rnn.get_input_size() == 10);
        ASSERT_TRUE(rnn.get_hidden_size() == 20);
        ASSERT_TRUE(rnn.is_trainable());
    }
    END_TEST("Construction - basic parameters");

    // ---------------------------------------------------------------
    TEST("Forward - output shape with return_sequences=true")
    {
        // Input: [batch=2, seq_len=5, input_size=10]
        // Output: [batch=2, seq_len=5, hidden_size=20]
        CppNet::Layers::RNN rnn(10, 20, true, "cpu-eigen");
        Eigen::Tensor<float, 3> input(2, 5, 10);
        input.setRandom();
        auto output = rnn.forward(input);

        ASSERT_TRUE(output.dimension(0) == 2);
        ASSERT_TRUE(output.dimension(1) == 5);
        ASSERT_TRUE(output.dimension(2) == 20);
    }
    END_TEST("Forward - output shape with return_sequences=true");

    // ---------------------------------------------------------------
    TEST("Forward - output values are in (-1, 1) due to tanh")
    {
        CppNet::Layers::RNN rnn(5, 8, true, "cpu-eigen");
        Eigen::Tensor<float, 3> input(1, 3, 5);
        input.setRandom();
        auto output = rnn.forward(input);

        for (int b = 0; b < output.dimension(0); ++b)
            for (int t = 0; t < output.dimension(1); ++t)
                for (int h = 0; h < output.dimension(2); ++h) {
                    ASSERT_TRUE(output(b, t, h) >= -1.0f);
                    ASSERT_TRUE(output(b, t, h) <= 1.0f);
                }
    }
    END_TEST("Forward - output values are in (-1, 1) due to tanh");

    // ---------------------------------------------------------------
    TEST("Forward - output is finite")
    {
        CppNet::Layers::RNN rnn(8, 16, true, "cpu-eigen");
        Eigen::Tensor<float, 3> input(2, 4, 8);
        input.setRandom();
        auto output = rnn.forward(input);

        for (int i = 0; i < output.size(); ++i)
            ASSERT_TRUE(std::isfinite(output.data()[i]));
    }
    END_TEST("Forward - output is finite");

    // ---------------------------------------------------------------
    TEST("Backward - grad input shape matches")
    {
        CppNet::Layers::RNN rnn(10, 20, true, "cpu-eigen");
        Eigen::Tensor<float, 3> input(2, 5, 10);
        input.setRandom();
        auto output = rnn.forward(input);

        Eigen::Tensor<float, 3> grad_output(output.dimensions());
        grad_output.setConstant(0.1f);
        auto grad_input = rnn.backward(grad_output);

        ASSERT_TRUE(grad_input.dimension(0) == 2);
        ASSERT_TRUE(grad_input.dimension(1) == 5);
        ASSERT_TRUE(grad_input.dimension(2) == 10);
    }
    END_TEST("Backward - grad input shape matches");

    // ---------------------------------------------------------------
    TEST("Backward - grad is finite")
    {
        CppNet::Layers::RNN rnn(5, 8, true, "cpu-eigen");
        Eigen::Tensor<float, 3> input(1, 3, 5);
        input.setRandom();
        auto output = rnn.forward(input);

        Eigen::Tensor<float, 3> grad_output(output.dimensions());
        grad_output.setConstant(0.01f);
        auto grad_input = rnn.backward(grad_output);

        for (int i = 0; i < grad_input.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad_input.data()[i]));
    }
    END_TEST("Backward - grad is finite");

    // ---------------------------------------------------------------
    TEST("Freeze/Unfreeze")
    {
        CppNet::Layers::RNN rnn(10, 20, true, "cpu-eigen");
        ASSERT_TRUE(rnn.is_trainable());
        rnn.freeze();
        ASSERT_TRUE(!rnn.is_trainable());
        rnn.unfreeze();
        ASSERT_TRUE(rnn.is_trainable());
    }
    END_TEST("Freeze/Unfreeze");

    // ---------------------------------------------------------------
    TEST("Single timestep")
    {
        CppNet::Layers::RNN rnn(4, 8, true, "cpu-eigen");
        Eigen::Tensor<float, 3> input(1, 1, 4);
        input.setRandom();
        auto output = rnn.forward(input);

        ASSERT_TRUE(output.dimension(0) == 1);
        ASSERT_TRUE(output.dimension(1) == 1);
        ASSERT_TRUE(output.dimension(2) == 8);
    }
    END_TEST("Single timestep");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
