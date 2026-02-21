/**
 * @file test_adam.cpp
 * @brief Unit tests for the Adam optimizer
 */

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/optimizers/adam.hpp"
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
    std::cout << "=== Adam Optimizer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Construction with default parameters")
    {
        CppNet::Optimizers::Adam adam;
        ASSERT_TRUE(true);
    }
    END_TEST("Construction with default parameters");

    // ---------------------------------------------------------------
    TEST("Construction with custom parameters")
    {
        CppNet::Optimizers::Adam adam(0.9f, 0.999f, 1e-8f);
        ASSERT_TRUE(true);
    }
    END_TEST("Construction with custom parameters");

    // ---------------------------------------------------------------
    TEST("Step updates weights")
    {
        CppNet::Layers::Linear layer(4, 2, "test", true, true, "cpu-eigen");
        CppNet::Optimizers::Adam adam;
        CppNet::Losses::MSE mse("mean");

        Eigen::Tensor<float, 2> original_weights = layer.get_weights();

        Eigen::Tensor<float, 2> input(1, 4);
        input.setRandom();
        auto output = layer.forward(input);

        Eigen::Tensor<float, 2> target(1, 2);
        target.setRandom();
        mse.forward(output, target);
        auto grad = mse.backward(output, target);
        layer.backward(grad);

        adam.step(layer, 0.001f);

        auto updated = layer.get_weights();
        bool changed = false;
        for (int i = 0; i < updated.size(); ++i) {
            if (std::fabs(updated.data()[i] - original_weights.data()[i]) > 1e-8f) {
                changed = true;
                break;
            }
        }
        ASSERT_TRUE(changed);
    }
    END_TEST("Step updates weights");

    // ---------------------------------------------------------------
    TEST("Multiple steps reduce loss")
    {
        CppNet::Layers::Linear layer(4, 2, "test", true, true, "cpu-eigen");
        CppNet::Optimizers::Adam adam;
        CppNet::Losses::MSE mse("mean");

        Eigen::Tensor<float, 2> input(2, 4);
        input.setRandom();
        Eigen::Tensor<float, 2> target(2, 2);
        target.setRandom();

        auto out = layer.forward(input);
        float initial_loss = mse.forward(out, target);

        for (int i = 0; i < 200; ++i) {
            auto output = layer.forward(input);
            mse.forward(output, target);
            auto grad = mse.backward(output, target);
            layer.backward(grad);
            adam.step(layer, 0.01f);
            layer.reset_grads();
        }

        auto final_out = layer.forward(input);
        float final_loss = mse.forward(final_out, target);
        ASSERT_TRUE(final_loss < initial_loss);
    }
    END_TEST("Multiple steps reduce loss");

    // ---------------------------------------------------------------
    TEST("Adam converges faster than SGD")
    {
        // Create two identical layers
        CppNet::Layers::Linear layer_adam(4, 2, "adam_layer", true, true, "cpu-eigen");
        CppNet::Layers::Linear layer_sgd(4, 2, "sgd_layer", true, true, "cpu-eigen");

        // Set identical initial weights
        layer_sgd.set_weights(layer_adam.get_weights());
        layer_sgd.set_biases(layer_adam.get_biases());

        CppNet::Optimizers::Adam adam;
        CppNet::Optimizers::SGD sgd;
        CppNet::Losses::MSE mse("mean");

        Eigen::Tensor<float, 2> input(4, 4);
        input.setRandom();
        Eigen::Tensor<float, 2> target(4, 2);
        target.setRandom();

        float lr = 0.01f;
        int steps = 50;

        for (int i = 0; i < steps; ++i) {
            // Adam step
            auto out_a = layer_adam.forward(input);
            mse.forward(out_a, target);
            auto grad_a = mse.backward(out_a, target);
            layer_adam.backward(grad_a);
            adam.step(layer_adam, lr);
            layer_adam.reset_grads();

            // SGD step
            auto out_s = layer_sgd.forward(input);
            mse.forward(out_s, target);
            auto grad_s = mse.backward(out_s, target);
            layer_sgd.backward(grad_s);
            sgd.step(layer_sgd, lr);
            layer_sgd.reset_grads();
        }

        auto final_a = layer_adam.forward(input);
        float loss_adam = mse.forward(final_a, target);
        auto final_s = layer_sgd.forward(input);
        float loss_sgd = mse.forward(final_s, target);

        // Adam should typically converge faster
        ASSERT_TRUE(loss_adam < loss_sgd || loss_adam < 0.05f);
    }
    END_TEST("Adam converges faster than SGD");

    // ---------------------------------------------------------------
    TEST("Weights remain finite after many steps")
    {
        CppNet::Layers::Linear layer(3, 2, "test", true, true, "cpu-eigen");
        CppNet::Optimizers::Adam adam;
        CppNet::Losses::MSE mse("mean");

        Eigen::Tensor<float, 2> input(1, 3);
        input.setRandom();
        Eigen::Tensor<float, 2> target(1, 2);
        target.setRandom();

        for (int i = 0; i < 100; ++i) {
            auto output = layer.forward(input);
            mse.forward(output, target);
            auto grad = mse.backward(output, target);
            layer.backward(grad);
            adam.step(layer, 0.001f);
            layer.reset_grads();
        }

        auto weights = layer.get_weights();
        for (int i = 0; i < weights.size(); ++i)
            ASSERT_TRUE(std::isfinite(weights.data()[i]));
    }
    END_TEST("Weights remain finite after many steps");

    // ---------------------------------------------------------------
    TEST("Frozen layer is not updated")
    {
        CppNet::Layers::Linear layer(3, 2, "frozen", true, true, "cpu-eigen");
        layer.freeze();
        CppNet::Optimizers::Adam adam;
        CppNet::Losses::MSE mse("mean");

        auto original = layer.get_weights();

        Eigen::Tensor<float, 2> input(1, 3);
        input.setRandom();
        auto output = layer.forward(input);
        Eigen::Tensor<float, 2> target(1, 2);
        target.setRandom();
        mse.forward(output, target);
        auto grad = mse.backward(output, target);
        layer.backward(grad);
        adam.step(layer, 0.01f);

        auto updated = layer.get_weights();
        for (int i = 0; i < updated.size(); ++i)
            ASSERT_NEAR(updated.data()[i], original.data()[i], 1e-7f);
    }
    END_TEST("Frozen layer is not updated");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
