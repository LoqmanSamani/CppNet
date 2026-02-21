/**
 * @file test_callbacks.cpp
 * @brief Unit tests for the EarlyStopping callback
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "CppNet/utils/callbacks.hpp"

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
    std::cout << "=== EarlyStopping Tests ===\n";

    TEST("Construction defaults") {
        CppNet::Callbacks::EarlyStopping es;
        ASSERT_TRUE(es.get_patience() == 5);
        ASSERT_TRUE(es.get_wait() == 0);
        ASSERT_TRUE(!es.should_stop());
    } END_TEST("Construction defaults");

    TEST("Custom parameters") {
        CppNet::Callbacks::EarlyStopping es(10, 0.01f, "max", false);
        ASSERT_TRUE(es.get_patience() == 10);
    } END_TEST("Custom parameters");

    TEST("Improving loss does not trigger") {
        CppNet::Callbacks::EarlyStopping es(3, 0.0f, "min", false);
        ASSERT_TRUE(!es.step(1.0f));
        ASSERT_TRUE(!es.step(0.9f));
        ASSERT_TRUE(!es.step(0.8f));
        ASSERT_TRUE(!es.should_stop());
    } END_TEST("Improving loss does not trigger");

    TEST("Stagnant loss triggers after patience") {
        CppNet::Callbacks::EarlyStopping es(3, 0.0f, "min", false);
        es.step(1.0f); // best
        es.step(1.1f); // wait=1
        es.step(1.2f); // wait=2
        bool stopped = es.step(1.3f); // wait=3 >= patience=3
        ASSERT_TRUE(stopped);
        ASSERT_TRUE(es.should_stop());
    } END_TEST("Stagnant loss triggers after patience");

    TEST("Improvement resets wait counter") {
        CppNet::Callbacks::EarlyStopping es(3, 0.0f, "min", false);
        es.step(1.0f);
        es.step(1.1f); // wait=1
        es.step(1.2f); // wait=2
        es.step(0.5f); // improved! wait resets
        ASSERT_TRUE(!es.should_stop());
        ASSERT_TRUE(es.get_wait() == 0);
    } END_TEST("Improvement resets wait counter");

    TEST("Mode 'max' - higher is better") {
        CppNet::Callbacks::EarlyStopping es(2, 0.0f, "max", false);
        es.step(0.5f);
        es.step(0.4f); // wait=1
        bool stopped = es.step(0.3f); // wait=2 >= patience=2
        ASSERT_TRUE(stopped);
    } END_TEST("Mode 'max' - higher is better");

    TEST("Reset clears state") {
        CppNet::Callbacks::EarlyStopping es(2, 0.0f, "min", false);
        es.step(1.0f);
        es.step(2.0f);
        es.step(3.0f); // stopped
        ASSERT_TRUE(es.should_stop());
        es.reset();
        ASSERT_TRUE(!es.should_stop());
        ASSERT_TRUE(es.get_wait() == 0);
    } END_TEST("Reset clears state");

    TEST("min_delta threshold") {
        CppNet::Callbacks::EarlyStopping es(2, 0.1f, "min", false);
        es.step(1.0f);
        // 0.95 is improvement but < min_delta (0.1)
        es.step(0.95f);
        bool stopped = es.step(0.92f);
        ASSERT_TRUE(stopped);
    } END_TEST("min_delta threshold");

    TEST("best_epoch tracks correctly") {
        CppNet::Callbacks::EarlyStopping es(5, 0.0f, "min", false);
        es.step(1.0f); // epoch 1, best
        es.step(0.5f); // epoch 2, new best
        es.step(0.8f); // epoch 3, no improvement
        ASSERT_TRUE(es.get_best_epoch() == 2);
    } END_TEST("best_epoch tracks correctly");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
