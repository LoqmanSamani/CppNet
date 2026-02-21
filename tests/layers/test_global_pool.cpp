/**
 * @file test_global_pool.cpp
 * @brief Unit tests for GlobalAvgPool2D and GlobalMaxPool2D
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/global_pool.hpp"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { std::cout << "  [TEST] " << name << " ... "; try {
#define END_TEST(name) std::cout << "PASSED\n"; ++tests_passed; \
    } catch (const std::exception& e) { std::cout << "FAILED: " << e.what() << "\n"; ++tests_failed; } \
      catch (...) { std::cout << "FAILED (unknown)\n"; ++tests_failed; } } while(0)
#define ASSERT_NEAR(a, b, eps) if (std::fabs((a)-(b))>(eps)) \
    throw std::runtime_error(std::string("Expected ")+std::to_string(b)+" got "+std::to_string(a))
#define ASSERT_TRUE(c) if (!(c)) throw std::runtime_error("Assertion failed: " #c)

int main() {
    std::cout << "=== Global Pool Tests ===\n";

    // --- GlobalAvgPool2D ---
    TEST("AvgPool: output shape [B,C,H,W] -> [B,C]") {
        CppNet::Layers::GlobalAvgPool2D pool;
        Eigen::Tensor<float, 4> input(2, 3, 4, 4);
        input.setRandom();
        auto out = pool.forward(input);
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(1) == 3);
    } END_TEST("AvgPool: output shape [B,C,H,W] -> [B,C]");

    TEST("AvgPool: known average") {
        CppNet::Layers::GlobalAvgPool2D pool;
        Eigen::Tensor<float, 4> input(1, 1, 2, 2);
        input.setValues({{{{1.0f, 3.0f}, {5.0f, 7.0f}}}});
        auto out = pool.forward(input);
        ASSERT_NEAR(out(0, 0), 4.0f, 1e-5f); // mean(1,3,5,7)=4
    } END_TEST("AvgPool: known average");

    TEST("AvgPool: not trainable") {
        CppNet::Layers::GlobalAvgPool2D pool;
        ASSERT_TRUE(!pool.is_trainable());
    } END_TEST("AvgPool: not trainable");

    TEST("AvgPool: backward shape") {
        CppNet::Layers::GlobalAvgPool2D pool;
        Eigen::Tensor<float, 4> input(2, 3, 4, 4);
        input.setRandom();
        pool.forward(input);
        Eigen::Tensor<float, 2> grad(2, 3);
        grad.setConstant(1.0f);
        auto grad_in = pool.backward(grad);
        ASSERT_TRUE(grad_in.dimension(0) == 2);
        ASSERT_TRUE(grad_in.dimension(1) == 3);
        ASSERT_TRUE(grad_in.dimension(2) == 4);
        ASSERT_TRUE(grad_in.dimension(3) == 4);
    } END_TEST("AvgPool: backward shape");

    TEST("AvgPool: backward distributes gradient equally") {
        CppNet::Layers::GlobalAvgPool2D pool;
        Eigen::Tensor<float, 4> input(1, 1, 2, 3);
        input.setConstant(1.0f);
        pool.forward(input);
        Eigen::Tensor<float, 2> grad(1, 1);
        grad.setConstant(6.0f);
        auto grad_in = pool.backward(grad);
        // Each element gets grad / (H*W) = 6 / 6 = 1
        for (int h = 0; h < 2; ++h)
            for (int w = 0; w < 3; ++w)
                ASSERT_NEAR(grad_in(0, 0, h, w), 1.0f, 1e-5f);
    } END_TEST("AvgPool: backward distributes gradient equally");

    // --- GlobalMaxPool2D ---
    TEST("MaxPool: output shape [B,C,H,W] -> [B,C]") {
        CppNet::Layers::GlobalMaxPool2D pool;
        Eigen::Tensor<float, 4> input(2, 3, 4, 4);
        input.setRandom();
        auto out = pool.forward(input);
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(1) == 3);
    } END_TEST("MaxPool: output shape [B,C,H,W] -> [B,C]");

    TEST("MaxPool: selects maximum") {
        CppNet::Layers::GlobalMaxPool2D pool;
        Eigen::Tensor<float, 4> input(1, 1, 2, 2);
        input.setValues({{{{1.0f, 3.0f}, {5.0f, 7.0f}}}});
        auto out = pool.forward(input);
        ASSERT_NEAR(out(0, 0), 7.0f, 1e-5f);
    } END_TEST("MaxPool: selects maximum");

    TEST("MaxPool: not trainable") {
        CppNet::Layers::GlobalMaxPool2D pool;
        ASSERT_TRUE(!pool.is_trainable());
    } END_TEST("MaxPool: not trainable");

    TEST("MaxPool: backward shape") {
        CppNet::Layers::GlobalMaxPool2D pool;
        Eigen::Tensor<float, 4> input(1, 2, 3, 3);
        input.setRandom();
        pool.forward(input);
        Eigen::Tensor<float, 2> grad(1, 2);
        grad.setConstant(1.0f);
        auto grad_in = pool.backward(grad);
        ASSERT_TRUE(grad_in.dimension(0) == 1);
        ASSERT_TRUE(grad_in.dimension(1) == 2);
        ASSERT_TRUE(grad_in.dimension(2) == 3);
        ASSERT_TRUE(grad_in.dimension(3) == 3);
    } END_TEST("MaxPool: backward shape");

    TEST("MaxPool: backward routes gradient to max location only") {
        CppNet::Layers::GlobalMaxPool2D pool;
        Eigen::Tensor<float, 4> input(1, 1, 2, 2);
        input.setValues({{{{1.0f, 3.0f}, {5.0f, 7.0f}}}});
        pool.forward(input);
        Eigen::Tensor<float, 2> grad(1, 1);
        grad.setConstant(1.0f);
        auto grad_in = pool.backward(grad);
        // Only the max element (7 at [1,1]) should get gradient
        ASSERT_NEAR(grad_in(0, 0, 0, 0), 0.0f, 1e-6f);
        ASSERT_NEAR(grad_in(0, 0, 0, 1), 0.0f, 1e-6f);
        ASSERT_NEAR(grad_in(0, 0, 1, 0), 0.0f, 1e-6f);
        ASSERT_NEAR(grad_in(0, 0, 1, 1), 1.0f, 1e-5f);
    } END_TEST("MaxPool: backward routes gradient to max location only");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
