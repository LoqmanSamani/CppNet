/**
 * @file test_huber.cpp
 * @brief Unit tests for the Huber loss
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/losses/huber.hpp"

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
    std::cout << "=== Huber Loss Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Perfect prediction gives zero loss")
    {
        CppNet::Losses::Huber huber(1.0f, "mean");
        Eigen::Tensor<float, 2> pred(2, 3);
        pred.setValues({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
        Eigen::Tensor<float, 2> target = pred;

        float loss = huber.forward(pred, target);
        ASSERT_NEAR(loss, 0.0f, 1e-6f);
    }
    END_TEST("Perfect prediction gives zero loss");

    // ---------------------------------------------------------------
    TEST("get_delta returns correct value")
    {
        CppNet::Losses::Huber huber(2.5f);
        ASSERT_NEAR(huber.get_delta(), 2.5f, 1e-6f);
    }
    END_TEST("get_delta returns correct value");

    // ---------------------------------------------------------------
    TEST("Small error: behaves like MSE")
    {
        // |pred - target| < delta → loss = 0.5 * (pred - target)^2
        float delta = 10.0f;
        CppNet::Losses::Huber huber(delta, "mean");
        Eigen::Tensor<float, 2> pred(1, 1);
        pred.setValues({{0.5f}});
        Eigen::Tensor<float, 2> target(1, 1);
        target.setValues({{0.0f}});

        float loss = huber.forward(pred, target);
        float expected = 0.5f * 0.5f * 0.5f; // 0.5 * 0.25 = 0.125
        ASSERT_NEAR(loss, expected, 1e-4f);
    }
    END_TEST("Small error: behaves like MSE");

    // ---------------------------------------------------------------
    TEST("Large error: behaves like MAE (linear)")
    {
        // |pred - target| > delta → loss = delta * (|err| - 0.5 * delta)
        float delta = 1.0f;
        CppNet::Losses::Huber huber(delta, "mean");
        Eigen::Tensor<float, 2> pred(1, 1);
        pred.setValues({{5.0f}});
        Eigen::Tensor<float, 2> target(1, 1);
        target.setValues({{0.0f}});

        float loss = huber.forward(pred, target);
        float expected = delta * (5.0f - 0.5f * delta); // 1 * (5 - 0.5) = 4.5
        ASSERT_NEAR(loss, expected, 0.1f);
    }
    END_TEST("Large error: behaves like MAE (linear)");

    // ---------------------------------------------------------------
    TEST("Loss is always non-negative")
    {
        CppNet::Losses::Huber huber(1.0f, "mean");
        Eigen::Tensor<float, 2> pred(4, 5);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(4, 5);
        target.setRandom();

        float loss = huber.forward(pred, target);
        ASSERT_TRUE(loss >= 0.0f);
    }
    END_TEST("Loss is always non-negative");

    // ---------------------------------------------------------------
    TEST("Backward - gradient shape matches")
    {
        CppNet::Losses::Huber huber(1.0f, "mean");
        Eigen::Tensor<float, 2> pred(2, 3);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(2, 3);
        target.setRandom();

        auto grad = huber.backward(pred, target);
        ASSERT_TRUE(grad.dimension(0) == 2);
        ASSERT_TRUE(grad.dimension(1) == 3);
    }
    END_TEST("Backward - gradient shape matches");

    // ---------------------------------------------------------------
    TEST("Backward - zero gradient when pred == target")
    {
        CppNet::Losses::Huber huber(1.0f, "mean");
        Eigen::Tensor<float, 2> pred(1, 3);
        pred.setValues({{1.0f, 2.0f, 3.0f}});
        Eigen::Tensor<float, 2> target = pred;

        auto grad = huber.backward(pred, target);
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(grad(0, j), 0.0f, 1e-5f);
    }
    END_TEST("Backward - zero gradient when pred == target");

    // ---------------------------------------------------------------
    TEST("Backward - gradient is finite")
    {
        CppNet::Losses::Huber huber(1.0f, "mean");
        Eigen::Tensor<float, 2> pred(3, 4);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(3, 4);
        target.setRandom();

        auto grad = huber.backward(pred, target);
        for (int i = 0; i < grad.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad.data()[i]));
    }
    END_TEST("Backward - gradient is finite");

    // ---------------------------------------------------------------
    TEST("Backward - gradient clipped for large errors")
    {
        float delta = 1.0f;
        CppNet::Losses::Huber huber(delta, "mean");
        Eigen::Tensor<float, 2> pred(1, 1);
        pred.setValues({{100.0f}});
        Eigen::Tensor<float, 2> target(1, 1);
        target.setValues({{0.0f}});

        auto grad = huber.backward(pred, target);
        // For large errors, gradient should be bounded by delta
        ASSERT_TRUE(std::fabs(grad(0, 0)) <= delta + 0.1f);
    }
    END_TEST("Backward - gradient clipped for large errors");

    // ---------------------------------------------------------------
    TEST("Different delta values")
    {
        Eigen::Tensor<float, 2> pred(1, 1);
        pred.setValues({{3.0f}});
        Eigen::Tensor<float, 2> target(1, 1);
        target.setValues({{0.0f}});

        CppNet::Losses::Huber huber_small(0.5f, "mean");
        CppNet::Losses::Huber huber_large(5.0f, "mean");

        float loss_small = huber_small.forward(pred, target);
        float loss_large = huber_large.forward(pred, target);

        // Both should be positive
        ASSERT_TRUE(loss_small > 0.0f);
        ASSERT_TRUE(loss_large > 0.0f);
    }
    END_TEST("Different delta values");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
