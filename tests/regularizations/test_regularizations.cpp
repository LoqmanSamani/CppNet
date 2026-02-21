/**
 * @file test_regularizations.cpp
 * @brief Unit tests for L1, L2, and ElasticNet regularization
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/regularizations/regularizations.hpp"

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
    std::cout << "=== Regularization Tests ===\n";

    // --- L1 ---
    TEST("l1_penalty: known value") {
        Eigen::Tensor<float, 2> w(2, 2);
        w.setValues({{1.0f, -2.0f}, {3.0f, -4.0f}});
        float p = CppNet::Regularizations::l1_penalty(w, 0.1f);
        // 0.1 * (1 + 2 + 3 + 4) = 0.1 * 10 = 1.0
        ASSERT_NEAR(p, 1.0f, 1e-5f);
    } END_TEST("l1_penalty: known value");

    TEST("l1_penalty: zero weights") {
        Eigen::Tensor<float, 2> w(3, 3);
        w.setZero();
        float p = CppNet::Regularizations::l1_penalty(w, 1.0f);
        ASSERT_NEAR(p, 0.0f, 1e-6f);
    } END_TEST("l1_penalty: zero weights");

    TEST("l1_gradient: sign function") {
        Eigen::Tensor<float, 2> w(1, 4);
        w.setValues({{2.0f, -3.0f, 0.0f, 5.0f}});
        auto g = CppNet::Regularizations::l1_gradient(w, 0.5f);
        ASSERT_NEAR(g(0, 0), 0.5f, 1e-5f);
        ASSERT_NEAR(g(0, 1), -0.5f, 1e-5f);
        // 0.0 could be 0 (sign 0)
        ASSERT_NEAR(g(0, 3), 0.5f, 1e-5f);
    } END_TEST("l1_gradient: sign function");

    TEST("l1_gradient: shape matches") {
        Eigen::Tensor<float, 2> w(3, 4);
        w.setRandom();
        auto g = CppNet::Regularizations::l1_gradient(w, 0.1f);
        ASSERT_TRUE(g.dimension(0) == 3);
        ASSERT_TRUE(g.dimension(1) == 4);
    } END_TEST("l1_gradient: shape matches");

    // --- L2 ---
    TEST("l2_penalty: known value") {
        Eigen::Tensor<float, 2> w(2, 2);
        w.setValues({{1.0f, 2.0f}, {3.0f, 4.0f}});
        float p = CppNet::Regularizations::l2_penalty(w, 0.1f);
        // 0.5 * 0.1 * (1 + 4 + 9 + 16) = 0.05 * 30 = 1.5
        ASSERT_NEAR(p, 1.5f, 1e-5f);
    } END_TEST("l2_penalty: known value");

    TEST("l2_penalty: zero weights") {
        Eigen::Tensor<float, 2> w(3, 3);
        w.setZero();
        float p = CppNet::Regularizations::l2_penalty(w, 1.0f);
        ASSERT_NEAR(p, 0.0f, 1e-6f);
    } END_TEST("l2_penalty: zero weights");

    TEST("l2_gradient: is lambda*W") {
        Eigen::Tensor<float, 2> w(2, 2);
        w.setValues({{1.0f, -2.0f}, {3.0f, -4.0f}});
        auto g = CppNet::Regularizations::l2_gradient(w, 0.5f);
        ASSERT_NEAR(g(0, 0), 0.5f, 1e-5f);
        ASSERT_NEAR(g(0, 1), -1.0f, 1e-5f);
        ASSERT_NEAR(g(1, 0), 1.5f, 1e-5f);
        ASSERT_NEAR(g(1, 1), -2.0f, 1e-5f);
    } END_TEST("l2_gradient: is lambda*W");

    // --- Elastic Net ---
    TEST("elastic_net_penalty: l1_ratio=1 equals L1") {
        Eigen::Tensor<float, 2> w(2, 2);
        w.setValues({{1.0f, -2.0f}, {3.0f, -4.0f}});
        float en = CppNet::Regularizations::elastic_net_penalty(w, 0.1f, 1.0f);
        float l1 = CppNet::Regularizations::l1_penalty(w, 0.1f);
        ASSERT_NEAR(en, l1, 1e-5f);
    } END_TEST("elastic_net_penalty: l1_ratio=1 equals L1");

    TEST("elastic_net_penalty: l1_ratio=0 equals L2") {
        Eigen::Tensor<float, 2> w(2, 2);
        w.setValues({{1.0f, 2.0f}, {3.0f, 4.0f}});
        float en = CppNet::Regularizations::elastic_net_penalty(w, 0.1f, 0.0f);
        float l2 = CppNet::Regularizations::l2_penalty(w, 0.1f);
        ASSERT_NEAR(en, l2, 1e-5f);
    } END_TEST("elastic_net_penalty: l1_ratio=0 equals L2");

    TEST("elastic_net_gradient: l1_ratio=0.5 combines") {
        Eigen::Tensor<float, 2> w(1, 2);
        w.setValues({{2.0f, -3.0f}});
        auto g = CppNet::Regularizations::elastic_net_gradient(w, 1.0f, 0.5f);
        // L1 part: 0.5 * sign = {0.5, -0.5}
        // L2 part: 0.5 * w = {1.0, -1.5}
        // Total: {1.5, -2.0}
        ASSERT_NEAR(g(0, 0), 1.5f, 1e-5f);
        ASSERT_NEAR(g(0, 1), -2.0f, 1e-5f);
    } END_TEST("elastic_net_gradient: l1_ratio=0.5 combines");

    TEST("elastic_net_gradient: shape matches") {
        Eigen::Tensor<float, 2> w(3, 4);
        w.setRandom();
        auto g = CppNet::Regularizations::elastic_net_gradient(w, 0.1f, 0.3f);
        ASSERT_TRUE(g.dimension(0) == 3);
        ASSERT_TRUE(g.dimension(1) == 4);
    } END_TEST("elastic_net_gradient: shape matches");

    TEST("Penalty non-negative for random weights") {
        Eigen::Tensor<float, 2> w(5, 5);
        w.setRandom();
        ASSERT_TRUE(CppNet::Regularizations::l1_penalty(w, 0.1f) >= 0.0f);
        ASSERT_TRUE(CppNet::Regularizations::l2_penalty(w, 0.1f) >= 0.0f);
        ASSERT_TRUE(CppNet::Regularizations::elastic_net_penalty(w, 0.1f, 0.5f) >= 0.0f);
    } END_TEST("Penalty non-negative for random weights");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
