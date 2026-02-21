/**
 * @file test_mse.cpp
 * @brief Unit tests for the MSE (Mean Squared Error) loss
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
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
    std::cout << "=== MSE Loss Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Perfect prediction gives zero loss")
    {
        CppNet::Losses::MSE mse("mean");
        Eigen::Tensor<float, 2> pred(2, 3);
        pred.setValues({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
        Eigen::Tensor<float, 2> target = pred;

        float loss = mse.forward(pred, target);
        ASSERT_NEAR(loss, 0.0f, 1e-6f);
    }
    END_TEST("Perfect prediction gives zero loss");

    // ---------------------------------------------------------------
    TEST("Known value: mean of squared errors")
    {
        CppNet::Losses::MSE mse("mean");
        Eigen::Tensor<float, 2> pred(1, 4);
        pred.setValues({{1.0f, 2.0f, 3.0f, 4.0f}});
        Eigen::Tensor<float, 2> target(1, 4);
        target.setValues({{1.0f, 2.0f, 3.0f, 4.0f}});

        float loss = mse.forward(pred, target);
        ASSERT_NEAR(loss, 0.0f, 1e-6f);
    }
    END_TEST("Known value: mean of squared errors");

    // ---------------------------------------------------------------
    TEST("Known value: non-zero MSE")
    {
        CppNet::Losses::MSE mse("mean");
        Eigen::Tensor<float, 2> pred(1, 3);
        pred.setValues({{0.0f, 0.0f, 0.0f}});
        Eigen::Tensor<float, 2> target(1, 3);
        target.setValues({{1.0f, 2.0f, 3.0f}});

        // MSE = mean(1^2 + 2^2 + 3^2) = mean(1 + 4 + 9) = 14/3 ≈ 4.6667
        float loss = mse.forward(pred, target);
        ASSERT_NEAR(loss, 14.0f / 3.0f, 0.1f);
    }
    END_TEST("Known value: non-zero MSE");

    // ---------------------------------------------------------------
    TEST("Loss is always non-negative")
    {
        CppNet::Losses::MSE mse("mean");
        Eigen::Tensor<float, 2> pred(3, 4);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(3, 4);
        target.setRandom();

        float loss = mse.forward(pred, target);
        ASSERT_TRUE(loss >= 0.0f);
    }
    END_TEST("Loss is always non-negative");

    // ---------------------------------------------------------------
    TEST("Backward - gradient shape matches")
    {
        CppNet::Losses::MSE mse("mean");
        Eigen::Tensor<float, 2> pred(2, 3);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(2, 3);
        target.setRandom();

        auto grad = mse.backward(pred, target);
        ASSERT_TRUE(grad.dimension(0) == 2);
        ASSERT_TRUE(grad.dimension(1) == 3);
    }
    END_TEST("Backward - gradient shape matches");

    // ---------------------------------------------------------------
    TEST("Backward - zero gradient when pred == target")
    {
        CppNet::Losses::MSE mse("mean");
        Eigen::Tensor<float, 2> pred(1, 3);
        pred.setValues({{1.0f, 2.0f, 3.0f}});
        Eigen::Tensor<float, 2> target = pred;

        auto grad = mse.backward(pred, target);
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(grad(0, j), 0.0f, 1e-6f);
    }
    END_TEST("Backward - zero gradient when pred == target");

    // ---------------------------------------------------------------
    TEST("Backward - gradient direction: pred > target → positive grad")
    {
        CppNet::Losses::MSE mse("mean");
        Eigen::Tensor<float, 2> pred(1, 2);
        pred.setValues({{5.0f, 10.0f}});
        Eigen::Tensor<float, 2> target(1, 2);
        target.setValues({{1.0f, 1.0f}});

        auto grad = mse.backward(pred, target);
        // dMSE/dpred = 2*(pred - target) / N → should be positive
        for (int j = 0; j < 2; ++j)
            ASSERT_TRUE(grad(0, j) > 0.0f);
    }
    END_TEST("Backward - gradient direction: pred > target → positive grad");

    // ---------------------------------------------------------------
    TEST("Backward - gradient is finite")
    {
        CppNet::Losses::MSE mse("mean");
        Eigen::Tensor<float, 2> pred(4, 5);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(4, 5);
        target.setRandom();

        auto grad = mse.backward(pred, target);
        for (int i = 0; i < grad.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad.data()[i]));
    }
    END_TEST("Backward - gradient is finite");

    // ---------------------------------------------------------------
    TEST("Symmetry: MSE(a, b) == MSE(b, a)")
    {
        CppNet::Losses::MSE mse("mean");
        Eigen::Tensor<float, 2> a(2, 3);
        a.setRandom();
        Eigen::Tensor<float, 2> b(2, 3);
        b.setRandom();

        float loss_ab = mse.forward(a, b);
        float loss_ba = mse.forward(b, a);
        ASSERT_NEAR(loss_ab, loss_ba, 1e-5f);
    }
    END_TEST("Symmetry: MSE(a, b) == MSE(b, a)");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
