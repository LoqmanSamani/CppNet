/**
 * @file test_utils.cpp
 * @brief Unit tests for general utility functions
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/utils/utils.hpp"

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
    std::cout << "=== Utils Tests ===\n";

    // --- one_hot_encode ---
    TEST("one_hot_encode: basic") {
        std::vector<int> labels = {0, 1, 2};
        auto oh = CppNet::Utils::one_hot_encode(labels, 3);
        ASSERT_TRUE(oh.dimension(0) == 3);
        ASSERT_TRUE(oh.dimension(1) == 3);
        ASSERT_NEAR(oh(0, 0), 1.0f, 1e-6f);
        ASSERT_NEAR(oh(0, 1), 0.0f, 1e-6f);
        ASSERT_NEAR(oh(1, 1), 1.0f, 1e-6f);
        ASSERT_NEAR(oh(2, 2), 1.0f, 1e-6f);
    } END_TEST("one_hot_encode: basic");

    TEST("one_hot_encode: auto-detect num_classes") {
        std::vector<int> labels = {1, 3, 0, 2};
        auto oh = CppNet::Utils::one_hot_encode(labels);
        ASSERT_TRUE(oh.dimension(0) == 4);
        ASSERT_TRUE(oh.dimension(1) >= 4); // should be at least 4 classes
        ASSERT_NEAR(oh(0, 1), 1.0f, 1e-6f);
        ASSERT_NEAR(oh(1, 3), 1.0f, 1e-6f);
    } END_TEST("one_hot_encode: auto-detect num_classes");

    // --- normalize ---
    TEST("normalize: zero mean per column") {
        Eigen::Tensor<float, 2> data(4, 2);
        data.setValues({{1, 10}, {2, 20}, {3, 30}, {4, 40}});
        auto n = CppNet::Utils::normalize(data);
        ASSERT_TRUE(n.dimension(0) == 4);
        ASSERT_TRUE(n.dimension(1) == 2);
        // Check that column means are near 0
        for (int col = 0; col < 2; ++col) {
            float sum = 0;
            for (int row = 0; row < 4; ++row) sum += n(row, col);
            ASSERT_NEAR(sum / 4.0f, 0.0f, 1e-4f);
        }
    } END_TEST("normalize: zero mean per column");

    TEST("normalize: unit variance per column") {
        Eigen::Tensor<float, 2> data(100, 1);
        for (int i = 0; i < 100; ++i) data(i, 0) = (float)i;
        auto n = CppNet::Utils::normalize(data);
        float sum2 = 0;
        for (int i = 0; i < 100; ++i) sum2 += n(i, 0) * n(i, 0);
        float var = sum2 / 100.0f;
        ASSERT_NEAR(var, 1.0f, 0.1f);
    } END_TEST("normalize: unit variance per column");

    // --- min_max_scale ---
    TEST("min_max_scale: output in [0,1]") {
        Eigen::Tensor<float, 2> data(4, 2);
        data.setValues({{1, 10}, {2, 20}, {3, 30}, {4, 40}});
        auto s = CppNet::Utils::min_max_scale(data);
        for (int i = 0; i < s.size(); ++i) {
            ASSERT_TRUE(s.data()[i] >= -1e-6f);
            ASSERT_TRUE(s.data()[i] <= 1.0f + 1e-6f);
        }
    } END_TEST("min_max_scale: output in [0,1]");

    TEST("min_max_scale: known values") {
        Eigen::Tensor<float, 2> data(3, 1);
        data.setValues({{0}, {5}, {10}});
        auto s = CppNet::Utils::min_max_scale(data);
        ASSERT_NEAR(s(0, 0), 0.0f, 1e-5f);
        ASSERT_NEAR(s(1, 0), 0.5f, 1e-5f);
        ASSERT_NEAR(s(2, 0), 1.0f, 1e-5f);
    } END_TEST("min_max_scale: known values");

    // --- shuffle_data ---
    TEST("shuffle_data: preserves dimensions") {
        Eigen::Tensor<float, 2> data(10, 3);
        Eigen::Tensor<float, 2> labels(10, 1);
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 3; ++j) data(i, j) = (float)(i * 3 + j);
            labels(i, 0) = (float)i;
        }
        CppNet::Utils::shuffle_data(data, labels, 42);
        ASSERT_TRUE(data.dimension(0) == 10);
        ASSERT_TRUE(data.dimension(1) == 3);
        ASSERT_TRUE(labels.dimension(0) == 10);
    } END_TEST("shuffle_data: preserves dimensions");

    TEST("shuffle_data: preserves total sum") {
        Eigen::Tensor<float, 2> data(10, 1);
        Eigen::Tensor<float, 2> labels(10, 1);
        for (int i = 0; i < 10; ++i) { data(i, 0) = (float)i; labels(i, 0) = (float)i; }
        float orig_sum = 45.0f;
        CppNet::Utils::shuffle_data(data, labels, 42);
        float new_sum = 0;
        for (int i = 0; i < 10; ++i) new_sum += labels(i, 0);
        ASSERT_NEAR(new_sum, orig_sum, 1e-4f);
    } END_TEST("shuffle_data: preserves total sum");

    // --- train_val_split ---
    TEST("train_val_split: correct sizes") {
        Eigen::Tensor<float, 2> data(100, 3);
        Eigen::Tensor<float, 2> labels(100, 1);
        data.setRandom();
        labels.setRandom();
        auto [td, tl, vd, vl] = CppNet::Utils::train_val_split(data, labels, 0.2f);
        ASSERT_TRUE(td.dimension(0) == 80);
        ASSERT_TRUE(vd.dimension(0) == 20);
        ASSERT_TRUE(tl.dimension(0) == 80);
        ASSERT_TRUE(vl.dimension(0) == 20);
    } END_TEST("train_val_split: correct sizes");

    TEST("train_val_split: feature dims preserved") {
        Eigen::Tensor<float, 2> data(50, 5);
        Eigen::Tensor<float, 2> labels(50, 2);
        data.setRandom();
        labels.setRandom();
        auto [td, tl, vd, vl] = CppNet::Utils::train_val_split(data, labels, 0.3f);
        ASSERT_TRUE(td.dimension(1) == 5);
        ASSERT_TRUE(tl.dimension(1) == 2);
        ASSERT_TRUE(vd.dimension(1) == 5);
        ASSERT_TRUE(vl.dimension(1) == 2);
    } END_TEST("train_val_split: feature dims preserved");

    // --- argmax ---
    TEST("argmax: basic") {
        Eigen::Tensor<float, 2> t(3, 4);
        t.setValues({{0.1f, 0.9f, 0.2f, 0.3f},
                     {0.5f, 0.1f, 0.3f, 0.8f},
                     {0.7f, 0.2f, 0.1f, 0.0f}});
        auto idx = CppNet::Utils::argmax(t);
        ASSERT_TRUE(idx.size() == 3);
        ASSERT_TRUE(idx[0] == 1);
        ASSERT_TRUE(idx[1] == 3);
        ASSERT_TRUE(idx[2] == 0);
    } END_TEST("argmax: basic");

    TEST("argmax: single element rows") {
        Eigen::Tensor<float, 2> t(2, 1);
        t.setValues({{5.0f}, {3.0f}});
        auto idx = CppNet::Utils::argmax(t);
        ASSERT_TRUE(idx[0] == 0);
        ASSERT_TRUE(idx[1] == 0);
    } END_TEST("argmax: single element rows");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
