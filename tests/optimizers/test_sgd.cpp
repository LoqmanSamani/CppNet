/**
 * @file test_sgd.cpp
 * @brief Unit tests for Stochastic Gradient Descent optimizer
 */

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/optimizers/sgd.hpp"
#include "CppNet/layers/linear.hpp"
#include "CppNet/losses/mse.hpp"

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
    std::cout << "=== SGD Optimizer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Construction")
    {
        CppNet::Optimizers::SGD sgd;
        ASSERT_TRUE(true); // No exception during construction
    }
    END_TEST("Construction");

    // ---------------------------------------------------------------
    TEST("Step updates weights")
    {
        CppNet::Layers::Linear layer(4, 2, "test_layer", true, true, "cpu-eigen");
        CppNet::Optimizers::SGD sgd;

        // Copy original weights
        Eigen::Tensor<float, 2> original_weights = layer.get_weights();

        // Create input and forward pass
        Eigen::Tensor<float, 2> input(1, 4);
        input.setRandom();
        auto output = layer.forward(input);

        // Create target and compute loss gradients
        Eigen::Tensor<float, 2> target(1, 2);
        target.setRandom();
        CppNet::Losses::MSE mse("mean");
        mse.forward(output, target);
        auto grad_output = mse.backward(output, target);

        // Backward through layer
        layer.backward(grad_output);

        // Apply SGD step
        sgd.step(layer, 0.01f);

        // Verify weights changed
        auto updated_weights = layer.get_weights();
        bool changed = false;
        for (int i = 0; i < updated_weights.size(); ++i) {
            if (std::fabs(updated_weights.data()[i] - original_weights.data()[i]) > 1e-8f) {
                changed = true;
                break;
            }
        }
        ASSERT_TRUE(changed);
    }
    END_TEST("Step updates weights");

    // ---------------------------------------------------------------
    TEST("Step updates biases")
    {
        CppNet::Layers::Linear layer(3, 2, "test_layer", true, true, "cpu-eigen");
        CppNet::Optimizers::SGD sgd;

        Eigen::Tensor<float, 1> original_biases = layer.get_biases();

        Eigen::Tensor<float, 2> input(1, 3);
        input.setRandom();
        auto output = layer.forward(input);

        Eigen::Tensor<float, 2> target(1, 2);
        target.setRandom();
        CppNet::Losses::MSE mse("mean");
        mse.forward(output, target);
        auto grad = mse.backward(output, target);
        layer.backward(grad);

        sgd.step(layer, 0.01f);

        Eigen::Tensor<float, 1> updated_biases = layer.get_biases();
        bool changed = false;
        for (int i = 0; i < updated_biases.size(); ++i) {
            if (std::fabs(updated_biases.data()[i] - original_biases.data()[i]) > 1e-8f) {
                changed = true;
                break;
            }
        }
        ASSERT_TRUE(changed);
    }
    END_TEST("Step updates biases");

    // ---------------------------------------------------------------
    TEST("Multiple steps reduce loss")
    {
        CppNet::Layers::Linear layer(4, 2, "test_layer", true, true, "cpu-eigen");
        CppNet::Optimizers::SGD sgd;
        CppNet::Losses::MSE mse("mean");

        Eigen::Tensor<float, 2> input(2, 4);
        input.setRandom();
        Eigen::Tensor<float, 2> target(2, 2);
        target.setRandom();

        // Initial loss
        auto out = layer.forward(input);
        float initial_loss = mse.forward(out, target);

        // Train for several steps
        for (int i = 0; i < 100; ++i) {
            auto output = layer.forward(input);
            mse.forward(output, target);
            auto grad = mse.backward(output, target);
            layer.backward(grad);
            sgd.step(layer, 0.01f);
            layer.reset_grads();
        }

        // Final loss should be lower
        auto final_out = layer.forward(input);
        float final_loss = mse.forward(final_out, target);
        ASSERT_TRUE(final_loss < initial_loss);
    }
    END_TEST("Multiple steps reduce loss");

    // ---------------------------------------------------------------
    TEST("Learning rate affects step size")
    {
        // Create two identical layers
        CppNet::Layers::Linear layer_small(3, 2, "small", true, true, "cpu-eigen");
        CppNet::Layers::Linear layer_large(3, 2, "large", true, true, "cpu-eigen");

        // Set identical weights
        auto weights = layer_small.get_weights();
        auto biases = layer_small.get_biases();
        layer_large.set_weights(weights);
        layer_large.set_biases(biases);

        CppNet::Optimizers::SGD sgd;
        CppNet::Losses::MSE mse("mean");

        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{1.0f, 2.0f, 3.0f}});
        Eigen::Tensor<float, 2> target(1, 2);
        target.setValues({{0.5f, 0.5f}});

        // Forward + backward on both layers
        auto out_s = layer_small.forward(input);
        mse.forward(out_s, target);
        auto grad_s = mse.backward(out_s, target);
        layer_small.backward(grad_s);

        auto out_l = layer_large.forward(input);
        mse.forward(out_l, target);
        auto grad_l = mse.backward(out_l, target);
        layer_large.backward(grad_l);

        // Step with different learning rates
        sgd.step(layer_small, 0.001f);
        sgd.step(layer_large, 0.1f);

        auto w_s = layer_small.get_weights();
        auto w_l = layer_large.get_weights();

        // Larger LR should cause larger weight change
        float change_small = 0.0f, change_large = 0.0f;
        for (int i = 0; i < weights.size(); ++i) {
            change_small += std::fabs(w_s.data()[i] - weights.data()[i]);
            change_large += std::fabs(w_l.data()[i] - weights.data()[i]);
        }
        ASSERT_TRUE(change_large > change_small);
    }
    END_TEST("Learning rate affects step size");

    // ---------------------------------------------------------------
    TEST("Frozen layer - weights don't change")
    {
        CppNet::Layers::Linear layer(3, 2, "frozen", true, true, "cpu-eigen");
        layer.freeze();
        CppNet::Optimizers::SGD sgd;

        Eigen::Tensor<float, 2> original_weights = layer.get_weights();

        Eigen::Tensor<float, 2> input(1, 3);
        input.setRandom();
        auto output = layer.forward(input);

        Eigen::Tensor<float, 2> target(1, 2);
        target.setRandom();
        CppNet::Losses::MSE mse("mean");
        mse.forward(output, target);
        auto grad = mse.backward(output, target);
        layer.backward(grad);
        sgd.step(layer, 0.01f);

        auto updated_weights = layer.get_weights();
        for (int i = 0; i < updated_weights.size(); ++i)
            ASSERT_NEAR(updated_weights.data()[i], original_weights.data()[i], 1e-7f);
    }
    END_TEST("Frozen layer - weights don't change");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
