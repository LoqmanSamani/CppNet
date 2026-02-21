/**
 * @file test_metrics.cpp
 * @brief Unit tests for CppNet evaluation metrics
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/metrics/metrics.hpp"

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
    std::cout << "=== Metrics Tests ===\n";

    // --- accuracy ---
    TEST("accuracy: perfect predictions") {
        Eigen::Tensor<float, 2> preds(3, 3);
        preds.setValues({{0.9f, 0.05f, 0.05f},
                         {0.1f, 0.8f, 0.1f},
                         {0.05f, 0.05f, 0.9f}});
        Eigen::Tensor<float, 2> targets(3, 3);
        targets.setValues({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        float acc = CppNet::Metrics::accuracy(preds, targets);
        ASSERT_NEAR(acc, 1.0f, 1e-5f);
    } END_TEST("accuracy: perfect predictions");

    TEST("accuracy: all wrong") {
        Eigen::Tensor<float, 2> preds(2, 3);
        preds.setValues({{0.1f, 0.1f, 0.8f},
                         {0.8f, 0.1f, 0.1f}});
        Eigen::Tensor<float, 2> targets(2, 3);
        targets.setValues({{1, 0, 0}, {0, 0, 1}});
        float acc = CppNet::Metrics::accuracy(preds, targets);
        ASSERT_NEAR(acc, 0.0f, 1e-5f);
    } END_TEST("accuracy: all wrong");

    TEST("accuracy: partial") {
        Eigen::Tensor<float, 2> preds(4, 2);
        preds.setValues({{0.9f, 0.1f}, {0.2f, 0.8f}, {0.6f, 0.4f}, {0.3f, 0.7f}});
        Eigen::Tensor<float, 2> targets(4, 2);
        targets.setValues({{1, 0}, {0, 1}, {0, 1}, {0, 1}});
        float acc = CppNet::Metrics::accuracy(preds, targets);
        ASSERT_NEAR(acc, 0.75f, 1e-5f); // 3/4 correct
    } END_TEST("accuracy: partial");

    // --- binary_accuracy ---
    TEST("binary_accuracy: perfect") {
        Eigen::Tensor<float, 2> preds(4, 1);
        preds.setValues({{0.9f}, {0.1f}, {0.8f}, {0.2f}});
        Eigen::Tensor<float, 2> targets(4, 1);
        targets.setValues({{1}, {0}, {1}, {0}});
        float acc = CppNet::Metrics::binary_accuracy(preds, targets);
        ASSERT_NEAR(acc, 1.0f, 1e-5f);
    } END_TEST("binary_accuracy: perfect");

    TEST("binary_accuracy: 50%") {
        Eigen::Tensor<float, 2> preds(4, 1);
        preds.setValues({{0.9f}, {0.9f}, {0.1f}, {0.1f}});
        Eigen::Tensor<float, 2> targets(4, 1);
        targets.setValues({{1}, {0}, {1}, {0}});
        float acc = CppNet::Metrics::binary_accuracy(preds, targets);
        ASSERT_NEAR(acc, 0.5f, 1e-5f);
    } END_TEST("binary_accuracy: 50%");

    TEST("binary_accuracy: custom threshold") {
        Eigen::Tensor<float, 2> preds(2, 1);
        preds.setValues({{0.6f}, {0.4f}});
        Eigen::Tensor<float, 2> targets(2, 1);
        targets.setValues({{1}, {0}});
        float acc = CppNet::Metrics::binary_accuracy(preds, targets, 0.7f);
        // pred 0.6 < 0.7 → 0 != 1 (wrong), pred 0.4 < 0.7 → 0 == 0 (right)
        ASSERT_NEAR(acc, 0.5f, 1e-5f);
    } END_TEST("binary_accuracy: custom threshold");

    // --- precision ---
    TEST("precision: all TP") {
        Eigen::Tensor<float, 2> preds(3, 1);
        preds.setValues({{0.9f}, {0.8f}, {0.7f}});
        Eigen::Tensor<float, 2> targets(3, 1);
        targets.setValues({{1}, {1}, {1}});
        float p = CppNet::Metrics::precision(preds, targets);
        ASSERT_NEAR(p, 1.0f, 1e-5f);
    } END_TEST("precision: all TP");

    TEST("precision: half FP") {
        Eigen::Tensor<float, 2> preds(4, 1);
        preds.setValues({{0.9f}, {0.8f}, {0.7f}, {0.6f}});
        Eigen::Tensor<float, 2> targets(4, 1);
        targets.setValues({{1}, {1}, {0}, {0}});
        float p = CppNet::Metrics::precision(preds, targets);
        ASSERT_NEAR(p, 0.5f, 1e-5f); // TP=2, FP=2
    } END_TEST("precision: half FP");

    // --- recall ---
    TEST("recall: all positives found") {
        Eigen::Tensor<float, 2> preds(4, 1);
        preds.setValues({{0.9f}, {0.8f}, {0.1f}, {0.2f}});
        Eigen::Tensor<float, 2> targets(4, 1);
        targets.setValues({{1}, {1}, {0}, {0}});
        float r = CppNet::Metrics::recall(preds, targets);
        ASSERT_NEAR(r, 1.0f, 1e-5f);
    } END_TEST("recall: all positives found");

    TEST("recall: half positives found") {
        Eigen::Tensor<float, 2> preds(4, 1);
        preds.setValues({{0.9f}, {0.2f}, {0.1f}, {0.2f}});
        Eigen::Tensor<float, 2> targets(4, 1);
        targets.setValues({{1}, {1}, {0}, {0}});
        float r = CppNet::Metrics::recall(preds, targets);
        ASSERT_NEAR(r, 0.5f, 1e-5f); // TP=1, FN=1
    } END_TEST("recall: half positives found");

    // --- f1_score ---
    TEST("f1_score: perfect") {
        Eigen::Tensor<float, 2> preds(4, 1);
        preds.setValues({{0.9f}, {0.1f}, {0.8f}, {0.2f}});
        Eigen::Tensor<float, 2> targets(4, 1);
        targets.setValues({{1}, {0}, {1}, {0}});
        float f1 = CppNet::Metrics::f1_score(preds, targets);
        ASSERT_NEAR(f1, 1.0f, 1e-5f);
    } END_TEST("f1_score: perfect");

    TEST("f1_score: known value") {
        // TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
        Eigen::Tensor<float, 2> preds(4, 1);
        preds.setValues({{0.9f}, {0.9f}, {0.1f}, {0.1f}});
        Eigen::Tensor<float, 2> targets(4, 1);
        targets.setValues({{1}, {0}, {1}, {0}});
        float f1 = CppNet::Metrics::f1_score(preds, targets);
        ASSERT_NEAR(f1, 0.5f, 1e-5f);
    } END_TEST("f1_score: known value");

    TEST("f1_score: range [0, 1]") {
        Eigen::Tensor<float, 2> preds(8, 1);
        preds.setRandom();
        Eigen::Tensor<float, 2> targets(8, 1);
        for (int i = 0; i < 8; ++i) targets(i, 0) = (float)(i % 2);
        float f1 = CppNet::Metrics::f1_score(preds, targets);
        ASSERT_TRUE(f1 >= 0.0f && f1 <= 1.0f);
    } END_TEST("f1_score: range [0, 1]");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
