/**
 * @file test_mae.cpp
 * @brief Unit tests for the MAE (Mean Absolute Error) loss
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/losses/mae.hpp"

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
    std::cout << "=== MAE Loss Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Perfect prediction gives zero loss")
    {
        CppNet::Losses::MAE mae("mean");
        Eigen::Tensor<float, 2> pred(2, 3);
        pred.setValues({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
        Eigen::Tensor<float, 2> target = pred;

        float loss = mae.forward(pred, target);
        ASSERT_NEAR(loss, 0.0f, 1e-6f);
    }
    END_TEST("Perfect prediction gives zero loss");

    // ---------------------------------------------------------------
    TEST("Known value: mean of absolute errors")
    {
        CppNet::Losses::MAE mae("mean");
        Eigen::Tensor<float, 2> pred(1, 3);
        pred.setValues({{0.0f, 0.0f, 0.0f}});
        Eigen::Tensor<float, 2> target(1, 3);
        target.setValues({{1.0f, 2.0f, 3.0f}});

        // MAE = mean(|1| + |2| + |3|) = 6/3 = 2.0
        float loss = mae.forward(pred, target);
        ASSERT_NEAR(loss, 2.0f, 0.1f);
    }
    END_TEST("Known value: mean of absolute errors");

    // ---------------------------------------------------------------
    TEST("Loss is always non-negative")
    {
        CppNet::Losses::MAE mae("mean");
        Eigen::Tensor<float, 2> pred(4, 5);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(4, 5);
        target.setRandom();

        float loss = mae.forward(pred, target);
        ASSERT_TRUE(loss >= 0.0f);
    }
    END_TEST("Loss is always non-negative");

    // ---------------------------------------------------------------
    TEST("Backward - gradient shape matches")
    {
        CppNet::Losses::MAE mae("mean");
        Eigen::Tensor<float, 2> pred(2, 3);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(2, 3);
        target.setRandom();

        auto grad = mae.backward(pred, target);
        ASSERT_TRUE(grad.dimension(0) == 2);
        ASSERT_TRUE(grad.dimension(1) == 3);
    }
    END_TEST("Backward - gradient shape matches");

    // ---------------------------------------------------------------
    TEST("Backward - gradient direction")
    {
        CppNet::Losses::MAE mae("mean");
        Eigen::Tensor<float, 2> pred(1, 2);
        pred.setValues({{5.0f, -5.0f}});
        Eigen::Tensor<float, 2> target(1, 2);
        target.setValues({{1.0f, 1.0f}});

        auto grad = mae.backward(pred, target);
        // dMAE/dpred = sign(pred - target) / N
        // pred > target → positive grad; pred < target → negative grad
        ASSERT_TRUE(grad(0, 0) > 0.0f);
        ASSERT_TRUE(grad(0, 1) < 0.0f);
    }
    END_TEST("Backward - gradient direction");

    // ---------------------------------------------------------------
    TEST("Backward - gradient is finite")
    {
        CppNet::Losses::MAE mae("mean");
        Eigen::Tensor<float, 2> pred(3, 4);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(3, 4);
        target.setRandom();

        auto grad = mae.backward(pred, target);
        for (int i = 0; i < grad.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad.data()[i]));
    }
    END_TEST("Backward - gradient is finite");

    // ---------------------------------------------------------------
    TEST("Symmetry: MAE(a, b) == MAE(b, a)")
    {
        CppNet::Losses::MAE mae("mean");
        Eigen::Tensor<float, 2> a(2, 3);
        a.setRandom();
        Eigen::Tensor<float, 2> b(2, 3);
        b.setRandom();

        float loss_ab = mae.forward(a, b);
        float loss_ba = mae.forward(b, a);
        ASSERT_NEAR(loss_ab, loss_ba, 1e-5f);
    }
    END_TEST("Symmetry: MAE(a, b) == MAE(b, a)");

    // ---------------------------------------------------------------
    TEST("MAE less sensitive to outliers than MSE")
    {
        CppNet::Losses::MAE mae("mean");
        Eigen::Tensor<float, 2> pred(1, 1);
        pred.setValues({{0.0f}});
        Eigen::Tensor<float, 2> target_small(1, 1);
        target_small.setValues({{1.0f}});
        Eigen::Tensor<float, 2> target_large(1, 1);
        target_large.setValues({{10.0f}});

        float loss_small = mae.forward(pred, target_small);
        float loss_large = mae.forward(pred, target_large);

        // MAE grows linearly: loss_large should be 10x loss_small
        ASSERT_NEAR(loss_large / loss_small, 10.0f, 0.5f);
    }
    END_TEST("MAE less sensitive to outliers than MSE");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
