/**
 * @file test_linear.cpp
 * @brief Unit tests for the Linear (fully connected) layer
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/linear.hpp"
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
    std::cout << "=== Linear Layer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Construction - basic parameters")
    {
        CppNet::Layers::Linear layer(10, 5, "fc1", true, true, "cpu-eigen", "xavier");
        ASSERT_TRUE(layer.get_input_size() == 10);
        ASSERT_TRUE(layer.get_output_size() == 5);
        ASSERT_TRUE(layer.get_layer_name() == "fc1");
        ASSERT_TRUE(layer.is_trainable());
        ASSERT_TRUE(layer.has_bias());
    }
    END_TEST("Construction - basic parameters");

    // ---------------------------------------------------------------
    TEST("Weight shape matches in_size x out_size")
    {
        CppNet::Layers::Linear layer(8, 4, "fc", true, true, "cpu-eigen", "xavier");
        auto& w = layer.get_weights();
        ASSERT_TRUE(w.dimension(0) == 8);
        ASSERT_TRUE(w.dimension(1) == 4);
    }
    END_TEST("Weight shape matches in_size x out_size");

    // ---------------------------------------------------------------
    TEST("Bias shape matches out_size")
    {
        CppNet::Layers::Linear layer(8, 4, "fc", true, true, "cpu-eigen", "xavier");
        auto& b = layer.get_biases();
        ASSERT_TRUE(b.dimension(0) == 4);
    }
    END_TEST("Bias shape matches out_size");

    // ---------------------------------------------------------------
    TEST("Forward - output shape is [batch, out_size]")
    {
        CppNet::Layers::Linear layer(10, 5, "fc", true, true, "cpu-eigen", "xavier");
        Eigen::Tensor<float, 2> input(3, 10);
        input.setRandom();
        auto output = layer.forward(input);

        ASSERT_TRUE(output.dimension(0) == 3);
        ASSERT_TRUE(output.dimension(1) == 5);
    }
    END_TEST("Forward - output shape is [batch, out_size]");

    // ---------------------------------------------------------------
    TEST("Forward - single sample")
    {
        CppNet::Layers::Linear layer(4, 2, "fc", true, true, "cpu-eigen", "xavier");
        Eigen::Tensor<float, 2> input(1, 4);
        input.setRandom();
        auto output = layer.forward(input);

        ASSERT_TRUE(output.dimension(0) == 1);
        ASSERT_TRUE(output.dimension(1) == 2);
        // Output should be finite
        for (int j = 0; j < 2; ++j)
            ASSERT_TRUE(std::isfinite(output(0, j)));
    }
    END_TEST("Forward - single sample");

    // ---------------------------------------------------------------
    TEST("Forward - manual computation check")
    {
        CppNet::Layers::Linear layer(2, 2, "fc", true, true, "cpu-eigen", "xavier");

        // Set known weights and biases
        Eigen::Tensor<float, 2> w(2, 2);
        w.setValues({{1.0f, 0.0f}, {0.0f, 1.0f}}); // identity
        layer.set_weights(w);

        Eigen::Tensor<float, 1> b(2);
        b.setValues({0.5f, -0.5f});
        layer.set_biases(b);

        Eigen::Tensor<float, 2> input(1, 2);
        input.setValues({{3.0f, 4.0f}});
        auto output = layer.forward(input);

        // output = input * W + bias = [3, 4] * I + [0.5, -0.5] = [3.5, 3.5]
        ASSERT_NEAR(output(0, 0), 3.5f, 1e-4f);
        ASSERT_NEAR(output(0, 1), 3.5f, 1e-4f);
    }
    END_TEST("Forward - manual computation check");

    // ---------------------------------------------------------------
    TEST("Forward without bias")
    {
        CppNet::Layers::Linear layer(3, 2, "fc", true, false, "cpu-eigen", "xavier");
        ASSERT_TRUE(!layer.has_bias());

        Eigen::Tensor<float, 2> input(2, 3);
        input.setRandom();
        auto output = layer.forward(input);

        ASSERT_TRUE(output.dimension(0) == 2);
        ASSERT_TRUE(output.dimension(1) == 2);
    }
    END_TEST("Forward without bias");

    // ---------------------------------------------------------------
    TEST("Backward - grad input shape matches input shape")
    {
        CppNet::Layers::Linear layer(6, 4, "fc", true, true, "cpu-eigen", "xavier");
        Eigen::Tensor<float, 2> input(2, 6);
        input.setRandom();
        layer.forward(input);

        Eigen::Tensor<float, 2> grad_output(2, 4);
        grad_output.setConstant(1.0f);
        auto grad_input = layer.backward(grad_output);

        ASSERT_TRUE(grad_input.dimension(0) == 2);
        ASSERT_TRUE(grad_input.dimension(1) == 6);
    }
    END_TEST("Backward - grad input shape matches input shape");

    // ---------------------------------------------------------------
    TEST("Backward - gradient weights shape")
    {
        CppNet::Layers::Linear layer(5, 3, "fc", true, true, "cpu-eigen", "xavier");
        Eigen::Tensor<float, 2> input(2, 5);
        input.setRandom();
        layer.forward(input);

        Eigen::Tensor<float, 2> grad_output(2, 3);
        grad_output.setConstant(1.0f);
        layer.backward(grad_output);

        auto& gw = layer.get_grad_weights();
        ASSERT_TRUE(gw.dimension(0) == 5);
        ASSERT_TRUE(gw.dimension(1) == 3);
    }
    END_TEST("Backward - gradient weights shape");

    // ---------------------------------------------------------------
    TEST("reset_grads clears gradients")
    {
        CppNet::Layers::Linear layer(4, 2, "fc", true, true, "cpu-eigen", "xavier");
        Eigen::Tensor<float, 2> input(1, 4);
        input.setRandom();
        layer.forward(input);

        Eigen::Tensor<float, 2> grad(1, 2);
        grad.setConstant(1.0f);
        layer.backward(grad);
        layer.reset_grads();

        auto& gw = layer.get_grad_weights();
        float sum = 0.0f;
        for (int i = 0; i < gw.dimension(0); ++i)
            for (int j = 0; j < gw.dimension(1); ++j)
                sum += std::fabs(gw(i, j));
        ASSERT_NEAR(sum, 0.0f, 1e-6f);
    }
    END_TEST("reset_grads clears gradients");

    // ---------------------------------------------------------------
    TEST("Freeze/Unfreeze")
    {
        CppNet::Layers::Linear layer(4, 2, "fc", true, true, "cpu-eigen", "xavier");
        ASSERT_TRUE(layer.is_trainable());

        layer.freeze();
        ASSERT_TRUE(!layer.is_trainable());

        layer.unfreeze();
        ASSERT_TRUE(layer.is_trainable());
    }
    END_TEST("Freeze/Unfreeze");

    // ---------------------------------------------------------------
    TEST("Step updates weights")
    {
        CppNet::Layers::Linear layer(4, 2, "fc", true, true, "cpu-eigen", "xavier");
        Eigen::Tensor<float, 2> input(1, 4);
        input.setConstant(1.0f);
        layer.forward(input);

        Eigen::Tensor<float, 2> grad(1, 2);
        grad.setConstant(1.0f);
        layer.backward(grad);

        // Save weights before step
        Eigen::Tensor<float, 2> w_before = layer.get_weights();

        CppNet::Optimizers::SGD optimizer;
        layer.step(optimizer, 0.01f);

        // Weights should have changed
        auto& w_after = layer.get_weights();
        bool changed = false;
        for (int i = 0; i < w_before.dimension(0) && !changed; ++i)
            for (int j = 0; j < w_before.dimension(1) && !changed; ++j)
                if (std::fabs(w_before(i, j) - w_after(i, j)) > 1e-8f)
                    changed = true;
        ASSERT_TRUE(changed);
    }
    END_TEST("Step updates weights");

    // ---------------------------------------------------------------
    TEST("CPU backend produces same shape")
    {
        CppNet::Layers::Linear layer(8, 4, "fc", true, true, "cpu", "xavier");
        Eigen::Tensor<float, 2> input(3, 8);
        input.setRandom();
        auto output = layer.forward(input);
        ASSERT_TRUE(output.dimension(0) == 3);
        ASSERT_TRUE(output.dimension(1) == 4);
    }
    END_TEST("CPU backend produces same shape");

    // ---------------------------------------------------------------
    TEST("set_weights / set_biases work correctly")
    {
        CppNet::Layers::Linear layer(3, 2, "fc", true, true, "cpu-eigen", "xavier");

        Eigen::Tensor<float, 2> new_w(3, 2);
        new_w.setConstant(0.5f);
        layer.set_weights(new_w);

        Eigen::Tensor<float, 1> new_b(2);
        new_b.setConstant(0.1f);
        layer.set_biases(new_b);

        auto& w = layer.get_weights();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 2; ++j)
                ASSERT_NEAR(w(i, j), 0.5f, 1e-6f);

        auto& b = layer.get_biases();
        for (int j = 0; j < 2; ++j)
            ASSERT_NEAR(b(j), 0.1f, 1e-6f);
    }
    END_TEST("set_weights / set_biases work correctly");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
