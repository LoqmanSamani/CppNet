/**
 * @file test_mrs_prop.cpp
 * @brief Unit tests for the RMSProp optimizer
 */

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/optimizers/mrs_prop.hpp"
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
    std::cout << "=== RMSProp Optimizer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Construction with defaults")
    {
        CppNet::Optimizers::RMSProp rmsprop;
        ASSERT_TRUE(true);
    }
    END_TEST("Construction with defaults");

    // ---------------------------------------------------------------
    TEST("Construction with custom parameters")
    {
        CppNet::Optimizers::RMSProp rmsprop(0.95f, 1e-6f);
        ASSERT_TRUE(true);
    }
    END_TEST("Construction with custom parameters");

    // ---------------------------------------------------------------
    TEST("Step updates weights")
    {
        CppNet::Layers::Linear layer(4, 2, "test", true, true, "cpu-eigen");
        CppNet::Optimizers::RMSProp rmsprop;
        CppNet::Losses::MSE mse("mean");

        auto original = layer.get_weights();

        Eigen::Tensor<float, 2> input(1, 4);
        input.setRandom();
        auto output = layer.forward(input);

        Eigen::Tensor<float, 2> target(1, 2);
        target.setRandom();
        mse.forward(output, target);
        auto grad = mse.backward(output, target);
        layer.backward(grad);

        rmsprop.step(layer, 0.01f);

        auto updated = layer.get_weights();
        bool changed = false;
        for (int i = 0; i < updated.size(); ++i) {
            if (std::fabs(updated.data()[i] - original.data()[i]) > 1e-8f) {
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
        CppNet::Optimizers::RMSProp rmsprop;
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
            rmsprop.step(layer, 0.01f);
            layer.reset_grads();
        }

        auto final_out = layer.forward(input);
        float final_loss = mse.forward(final_out, target);
        ASSERT_TRUE(final_loss < initial_loss);
    }
    END_TEST("Multiple steps reduce loss");

    // ---------------------------------------------------------------
    TEST("RMSProp adapts learning rate per-parameter")
    {
        // Train on a simple task and verify convergence
        CppNet::Layers::Linear layer(2, 1, "test", true, true, "cpu-eigen");
        CppNet::Optimizers::RMSProp rmsprop;
        CppNet::Losses::MSE mse("mean");

        // Simple data: y = x1 + x2
        Eigen::Tensor<float, 2> input(4, 2);
        input.setValues({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
        Eigen::Tensor<float, 2> target(4, 1);
        target.setValues({{3}, {7}, {11}, {15}});

        for (int i = 0; i < 1000; ++i) {
            auto output = layer.forward(input);
            mse.forward(output, target);
            auto grad = mse.backward(output, target);
            layer.backward(grad);
            rmsprop.step(layer, 0.001f);
            layer.reset_grads();
        }

        // Should produce reasonable predictions
        auto final_out = layer.forward(input);
        float final_loss = mse.forward(final_out, target);
        ASSERT_TRUE(final_loss < 10.0f);
    }
    END_TEST("RMSProp adapts learning rate per-parameter");

    // ---------------------------------------------------------------
    TEST("Weights remain finite")
    {
        CppNet::Layers::Linear layer(3, 2, "test", true, true, "cpu-eigen");
        CppNet::Optimizers::RMSProp rmsprop;
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
            rmsprop.step(layer, 0.001f);
            layer.reset_grads();
        }

        auto weights = layer.get_weights();
        for (int i = 0; i < weights.size(); ++i)
            ASSERT_TRUE(std::isfinite(weights.data()[i]));
    }
    END_TEST("Weights remain finite");

    // ---------------------------------------------------------------
    TEST("Frozen layer not updated")
    {
        CppNet::Layers::Linear layer(3, 2, "frozen", true, true, "cpu-eigen");
        layer.freeze();
        CppNet::Optimizers::RMSProp rmsprop;
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
        rmsprop.step(layer, 0.01f);

        auto updated = layer.get_weights();
        for (int i = 0; i < updated.size(); ++i)
            ASSERT_NEAR(updated.data()[i], original.data()[i], 1e-7f);
    }
    END_TEST("Frozen layer not updated");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
