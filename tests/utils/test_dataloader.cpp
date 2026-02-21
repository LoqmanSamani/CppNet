/**
 * @file test_dataloader.cpp
 * @brief Unit tests for the DataLoader
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/utils/dataloader.hpp"

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
    std::cout << "=== DataLoader Tests ===\n";

    // Prepare a small dataset
    Eigen::Tensor<float, 2> features(10, 3);
    Eigen::Tensor<float, 2> labels(10, 1);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 3; ++j) features(i, j) = (float)(i * 3 + j);
        labels(i, 0) = (float)i;
    }

    TEST("Construction") {
        CppNet::Utils::DataLoader loader(features, labels, 4, false, false);
        ASSERT_TRUE(loader.num_samples() == 10);
        ASSERT_TRUE(loader.get_batch_size() == 4);
    } END_TEST("Construction");

    TEST("num_batches without drop_last") {
        CppNet::Utils::DataLoader loader(features, labels, 4, false, false);
        // 10 / 4 = 2 full + 1 partial = 3
        ASSERT_TRUE(loader.num_batches() == 3);
    } END_TEST("num_batches without drop_last");

    TEST("num_batches with drop_last") {
        CppNet::Utils::DataLoader loader(features, labels, 4, false, true);
        // 10 / 4 = 2 (drop last incomplete batch)
        ASSERT_TRUE(loader.num_batches() == 2);
    } END_TEST("num_batches with drop_last");

    TEST("Iterator yields correct number of batches") {
        CppNet::Utils::DataLoader loader(features, labels, 3, false, false);
        int count = 0;
        for (auto it = loader.begin(); it != loader.end(); ++it) {
            ++count;
        }
        ASSERT_TRUE(count == 4); // ceil(10/3) = 4
    } END_TEST("Iterator yields correct number of batches");

    TEST("Batch dimensions") {
        CppNet::Utils::DataLoader loader(features, labels, 4, false, false);
        auto it = loader.begin();
        auto batch = *it;
        ASSERT_TRUE(batch.features.dimension(0) == 4);
        ASSERT_TRUE(batch.features.dimension(1) == 3);
        ASSERT_TRUE(batch.labels.dimension(0) == 4);
        ASSERT_TRUE(batch.labels.dimension(1) == 1);
    } END_TEST("Batch dimensions");

    TEST("Last batch size without drop_last") {
        CppNet::Utils::DataLoader loader(features, labels, 4, false, false);
        CppNet::Utils::Batch last_batch;
        for (auto batch : loader) last_batch = batch;
        // Last batch has 10 - 2*4 = 2 samples
        ASSERT_TRUE(last_batch.features.dimension(0) == 2);
    } END_TEST("Last batch size without drop_last");

    TEST("All data covered without shuffle") {
        CppNet::Utils::DataLoader loader(features, labels, 10, false, false);
        auto it = loader.begin();
        auto batch = *it;
        // Should be entire dataset in one batch
        ASSERT_TRUE(batch.features.dimension(0) == 10);
        for (int i = 0; i < 10; ++i)
            ASSERT_NEAR(batch.labels(i, 0), (float)i, 1e-5f);
    } END_TEST("All data covered without shuffle");

    TEST("Reset allows re-iteration") {
        CppNet::Utils::DataLoader loader(features, labels, 5, false, false);
        int count1 = 0;
        for (auto b : loader) ++count1;
        loader.reset();
        int count2 = 0;
        for (auto b : loader) ++count2;
        ASSERT_TRUE(count1 == count2);
    } END_TEST("Reset allows re-iteration");

    TEST("Shuffle does not lose data") {
        CppNet::Utils::DataLoader loader(features, labels, 10, true, false);
        auto it = loader.begin();
        auto batch = *it;
        // Sum of labels should be same regardless of shuffle: 0+1+...+9 = 45
        float sum = 0;
        for (int i = 0; i < 10; ++i) sum += batch.labels(i, 0);
        ASSERT_NEAR(sum, 45.0f, 1e-3f);
    } END_TEST("Shuffle does not lose data");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
