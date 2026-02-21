/**
 * @file test_init.cpp
 * @brief Unit tests for weight initialization strategies
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/utils/init.hpp"

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
    std::cout << "=== Init Tests ===\n";

    TEST("xavier_uniform: shape") {
        auto w = CppNet::Utils::xavier_uniform(8, 16);
        ASSERT_TRUE(w.dimension(0) == 8);
        ASSERT_TRUE(w.dimension(1) == 16);
    } END_TEST("xavier_uniform: shape");

    TEST("xavier_uniform: bounded by sqrt(6/(fan_in+fan_out))") {
        auto w = CppNet::Utils::xavier_uniform(100, 200);
        float limit = std::sqrt(6.0f / (100 + 200));
        for (int i = 0; i < w.size(); ++i) {
            ASSERT_TRUE(w.data()[i] >= -limit - 1e-5f);
            ASSERT_TRUE(w.data()[i] <= limit + 1e-5f);
        }
    } END_TEST("xavier_uniform: bounded by sqrt(6/(fan_in+fan_out))");

    TEST("xavier_normal: shape") {
        auto w = CppNet::Utils::xavier_normal(8, 16);
        ASSERT_TRUE(w.dimension(0) == 8);
        ASSERT_TRUE(w.dimension(1) == 16);
    } END_TEST("xavier_normal: shape");

    TEST("xavier_normal: mean near zero") {
        auto w = CppNet::Utils::xavier_normal(200, 200);
        float sum = 0;
        for (int i = 0; i < w.size(); ++i) sum += w.data()[i];
        float mean = sum / (float)w.size();
        ASSERT_NEAR(mean, 0.0f, 0.05f);
    } END_TEST("xavier_normal: mean near zero");

    TEST("he_uniform: shape") {
        auto w = CppNet::Utils::he_uniform(10, 20);
        ASSERT_TRUE(w.dimension(0) == 10);
        ASSERT_TRUE(w.dimension(1) == 20);
    } END_TEST("he_uniform: shape");

    TEST("he_uniform: bounded by sqrt(6/fan_in)") {
        auto w = CppNet::Utils::he_uniform(100, 200);
        float limit = std::sqrt(6.0f / 100);
        for (int i = 0; i < w.size(); ++i) {
            ASSERT_TRUE(w.data()[i] >= -limit - 1e-5f);
            ASSERT_TRUE(w.data()[i] <= limit + 1e-5f);
        }
    } END_TEST("he_uniform: bounded by sqrt(6/fan_in)");

    TEST("he_normal: shape") {
        auto w = CppNet::Utils::he_normal(10, 20);
        ASSERT_TRUE(w.dimension(0) == 10);
        ASSERT_TRUE(w.dimension(1) == 20);
    } END_TEST("he_normal: shape");

    TEST("uniform_init: within [low, high)") {
        auto w = CppNet::Utils::uniform_init(50, 50, -2.0f, 2.0f);
        for (int i = 0; i < w.size(); ++i) {
            ASSERT_TRUE(w.data()[i] >= -2.0f - 1e-5f);
            ASSERT_TRUE(w.data()[i] <= 2.0f + 1e-5f);
        }
    } END_TEST("uniform_init: within [low, high)");

    TEST("normal_init: mean near zero, shape correct") {
        auto w = CppNet::Utils::normal_init(100, 100, 0.0f, 1.0f);
        ASSERT_TRUE(w.dimension(0) == 100);
        ASSERT_TRUE(w.dimension(1) == 100);
        float sum = 0;
        for (int i = 0; i < w.size(); ++i) sum += w.data()[i];
        float mean = sum / (float)w.size();
        ASSERT_NEAR(mean, 0.0f, 0.1f);
    } END_TEST("normal_init: mean near zero, shape correct");

    TEST("constant_init: all same value") {
        auto w = CppNet::Utils::constant_init(3, 4, 7.5f);
        for (int i = 0; i < w.size(); ++i)
            ASSERT_NEAR(w.data()[i], 7.5f, 1e-6f);
    } END_TEST("constant_init: all same value");

    TEST("constant_init: zeros by default") {
        auto w = CppNet::Utils::constant_init(2, 2);
        for (int i = 0; i < w.size(); ++i)
            ASSERT_NEAR(w.data()[i], 0.0f, 1e-6f);
    } END_TEST("constant_init: zeros by default");

    TEST("init_weights: 'xavier' strategy") {
        auto w = CppNet::Utils::init_weights(8, 16, "xavier");
        ASSERT_TRUE(w.dimension(0) == 8);
        ASSERT_TRUE(w.dimension(1) == 16);
    } END_TEST("init_weights: 'xavier' strategy");

    TEST("init_weights: 'he' strategy") {
        auto w = CppNet::Utils::init_weights(8, 16, "he");
        ASSERT_TRUE(w.dimension(0) == 8);
        ASSERT_TRUE(w.dimension(1) == 16);
    } END_TEST("init_weights: 'he' strategy");

    TEST("init_weights: 'zeros' strategy") {
        auto w = CppNet::Utils::init_weights(3, 3, "zeros");
        for (int i = 0; i < w.size(); ++i)
            ASSERT_NEAR(w.data()[i], 0.0f, 1e-6f);
    } END_TEST("init_weights: 'zeros' strategy");

    TEST("All values finite") {
        auto w1 = CppNet::Utils::xavier_uniform(50, 50);
        auto w2 = CppNet::Utils::he_normal(50, 50);
        for (int i = 0; i < w1.size(); ++i) ASSERT_TRUE(std::isfinite(w1.data()[i]));
        for (int i = 0; i < w2.size(); ++i) ASSERT_TRUE(std::isfinite(w2.data()[i]));
    } END_TEST("All values finite");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
