/**
 * @file test_momentum.cpp
 * @brief Unit tests for the Momentum (SGD with momentum) optimizer
 */

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/optimizers/momentum.hpp"
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
    std::cout << "=== Momentum Optimizer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Construction with default mu")
    {
        CppNet::Optimizers::Momentum momentum;
        ASSERT_TRUE(true);
    }
    END_TEST("Construction with default mu");

    // ---------------------------------------------------------------
    TEST("Construction with custom mu")
    {
        CppNet::Optimizers::Momentum momentum(0.99f);
        ASSERT_TRUE(true);
    }
    END_TEST("Construction with custom mu");

    // ---------------------------------------------------------------
    TEST("Step updates weights")
    {
        CppNet::Layers::Linear layer(4, 2, "test", true, true, "cpu-eigen");
        CppNet::Optimizers::Momentum momentum;
        CppNet::Losses::MSE mse("mean");

        auto original_weights = layer.get_weights();

        Eigen::Tensor<float, 2> input(1, 4);
        input.setRandom();
        auto output = layer.forward(input);

        Eigen::Tensor<float, 2> target(1, 2);
        target.setRandom();
        mse.forward(output, target);
        auto grad = mse.backward(output, target);
        layer.backward(grad);

        momentum.step(layer, 0.01f);

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
        CppNet::Optimizers::Momentum momentum;
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
            momentum.step(layer, 0.01f);
            layer.reset_grads();
        }

        auto final_out = layer.forward(input);
        float final_loss = mse.forward(final_out, target);
        ASSERT_TRUE(final_loss < initial_loss);
    }
    END_TEST("Multiple steps reduce loss");

    // ---------------------------------------------------------------
    TEST("Momentum builds velocity over consistent gradients")
    {
        CppNet::Layers::Linear layer(3, 1, "test", true, true, "cpu-eigen");
        CppNet::Optimizers::Momentum momentum(0.9f);
        CppNet::Losses::MSE mse("mean");

        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{1.0f, 1.0f, 1.0f}});
        Eigen::Tensor<float, 2> target(1, 1);
        target.setValues({{0.0f}});

        float lr = 0.01f;

        // First step - no accumulated velocity
        auto out1 = layer.forward(input);
        mse.forward(out1, target);
        auto grad1 = mse.backward(out1, target);
        layer.backward(grad1);
        auto w_before1 = layer.get_weights();
        momentum.step(layer, lr);
        auto w_after1 = layer.get_weights();
        layer.reset_grads();

        float delta1 = 0.0f;
        for (int i = 0; i < w_before1.size(); ++i)
            delta1 += std::fabs(w_after1.data()[i] - w_before1.data()[i]);

        // Second step - should have accumulated velocity
        auto out2 = layer.forward(input);
        mse.forward(out2, target);
        auto grad2 = mse.backward(out2, target);
        layer.backward(grad2);
        auto w_before2 = layer.get_weights();
        momentum.step(layer, lr);
        auto w_after2 = layer.get_weights();

        float delta2 = 0.0f;
        for (int i = 0; i < w_before2.size(); ++i)
            delta2 += std::fabs(w_after2.data()[i] - w_before2.data()[i]);

        // With momentum, second step should be at least as large (velocity builds up)
        ASSERT_TRUE(delta2 >= delta1 * 0.9f);
    }
    END_TEST("Momentum builds velocity over consistent gradients");

    // ---------------------------------------------------------------
    TEST("Weights remain finite")
    {
        CppNet::Layers::Linear layer(3, 2, "test", true, true, "cpu-eigen");
        CppNet::Optimizers::Momentum momentum;
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
            momentum.step(layer, 0.01f);
            layer.reset_grads();
        }

        auto weights = layer.get_weights();
        for (int i = 0; i < weights.size(); ++i)
            ASSERT_TRUE(std::isfinite(weights.data()[i]));
    }
    END_TEST("Weights remain finite");

    // ---------------------------------------------------------------
    TEST("Frozen layer - weights unchanged")
    {
        CppNet::Layers::Linear layer(3, 2, "frozen", true, true, "cpu-eigen");
        layer.freeze();
        CppNet::Optimizers::Momentum momentum;
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
        momentum.step(layer, 0.01f);

        auto updated = layer.get_weights();
        for (int i = 0; i < updated.size(); ++i)
            ASSERT_NEAR(updated.data()[i], original.data()[i], 1e-7f);
    }
    END_TEST("Frozen layer - weights unchanged");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
