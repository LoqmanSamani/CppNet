/**
 * @file test_gradient_clip.cpp
 * @brief Unit tests for gradient clipping utilities
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/utils/gradient_clip.hpp"

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
    std::cout << "=== Gradient Clip Tests ===\n";

    // --- clip_by_value (2D) ---
    TEST("clip_by_value 2D: clamps elements") {
        Eigen::Tensor<float, 2> t(2, 3);
        t.setValues({{-5.0f, 0.5f, 3.0f}, {-0.1f, 2.0f, -4.0f}});
        CppNet::Utils::clip_by_value(t, 1.0f);
        for (int i = 0; i < t.size(); ++i) {
            ASSERT_TRUE(t.data()[i] >= -1.0f);
            ASSERT_TRUE(t.data()[i] <= 1.0f);
        }
    } END_TEST("clip_by_value 2D: clamps elements");

    TEST("clip_by_value 2D: no change if within bounds") {
        Eigen::Tensor<float, 2> t(1, 3);
        t.setValues({{-0.5f, 0.0f, 0.5f}});
        CppNet::Utils::clip_by_value(t, 1.0f);
        ASSERT_NEAR(t(0, 0), -0.5f, 1e-6f);
        ASSERT_NEAR(t(0, 1), 0.0f, 1e-6f);
        ASSERT_NEAR(t(0, 2), 0.5f, 1e-6f);
    } END_TEST("clip_by_value 2D: no change if within bounds");

    // --- clip_by_value (1D) ---
    TEST("clip_by_value 1D: clamps elements") {
        Eigen::Tensor<float, 1> t(4);
        t.setValues({-10.0f, -1.0f, 1.0f, 10.0f});
        CppNet::Utils::clip_by_value(t, 2.0f);
        ASSERT_NEAR(t(0), -2.0f, 1e-6f);
        ASSERT_NEAR(t(1), -1.0f, 1e-6f);
        ASSERT_NEAR(t(2), 1.0f, 1e-6f);
        ASSERT_NEAR(t(3), 2.0f, 1e-6f);
    } END_TEST("clip_by_value 1D: clamps elements");

    // --- clip_by_norm (2D) ---
    TEST("clip_by_norm 2D: scales down when norm exceeds") {
        Eigen::Tensor<float, 2> t(1, 3);
        t.setValues({{3.0f, 4.0f, 0.0f}}); // norm = 5
        float orig_norm = CppNet::Utils::clip_by_norm(t, 1.0f);
        ASSERT_NEAR(orig_norm, 5.0f, 1e-4f);
        // After clipping, norm should be ≤ 1
        float new_norm = 0;
        for (int i = 0; i < t.size(); ++i) new_norm += t.data()[i] * t.data()[i];
        new_norm = std::sqrt(new_norm);
        ASSERT_NEAR(new_norm, 1.0f, 1e-4f);
    } END_TEST("clip_by_norm 2D: scales down when norm exceeds");

    TEST("clip_by_norm 2D: no change when norm under limit") {
        Eigen::Tensor<float, 2> t(1, 2);
        t.setValues({{0.3f, 0.4f}}); // norm = 0.5
        float orig = CppNet::Utils::clip_by_norm(t, 10.0f);
        ASSERT_NEAR(orig, 0.5f, 1e-4f);
        ASSERT_NEAR(t(0, 0), 0.3f, 1e-6f);
        ASSERT_NEAR(t(0, 1), 0.4f, 1e-6f);
    } END_TEST("clip_by_norm 2D: no change when norm under limit");

    // --- clip_by_norm (1D) ---
    TEST("clip_by_norm 1D: scales correctly") {
        Eigen::Tensor<float, 1> t(2);
        t.setValues({3.0f, 4.0f}); // norm = 5
        float orig = CppNet::Utils::clip_by_norm(t, 2.5f);
        ASSERT_NEAR(orig, 5.0f, 1e-4f);
        float new_norm = std::sqrt(t(0)*t(0) + t(1)*t(1));
        ASSERT_NEAR(new_norm, 2.5f, 1e-4f);
    } END_TEST("clip_by_norm 1D: scales correctly");

    TEST("clip_by_norm 1D: preserves direction") {
        Eigen::Tensor<float, 1> t(2);
        t.setValues({6.0f, 8.0f}); // norm = 10
        CppNet::Utils::clip_by_norm(t, 5.0f);
        // Direction: 6/10, 8/10 → 3, 4
        ASSERT_NEAR(t(0), 3.0f, 1e-4f);
        ASSERT_NEAR(t(1), 4.0f, 1e-4f);
    } END_TEST("clip_by_norm 1D: preserves direction");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
