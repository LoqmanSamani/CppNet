/**
 * @file test_elapsed_time.cpp
 * @brief Unit tests for the Timer utility
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <chrono>
#include "CppNet/utils/elapsed_time.hpp"

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
    std::cout << "=== Timer Tests ===\n";

    TEST("Construction") {
        CppNet::Utils::Timer t("test", false);
        ASSERT_TRUE(true); // no throw
    } END_TEST("Construction");

    TEST("Elapsed is non-negative") {
        CppNet::Utils::Timer t("", false);
        ASSERT_TRUE(t.elapsed_seconds() >= 0.0);
        ASSERT_TRUE(t.elapsed_ms() >= 0.0);
    } END_TEST("Elapsed is non-negative");

    TEST("Elapsed increases over time") {
        CppNet::Utils::Timer t("", false);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        double ms = t.elapsed_ms();
        ASSERT_TRUE(ms >= 10.0); // at least ~10ms (generous margin)
    } END_TEST("Elapsed increases over time");

    TEST("elapsed_ms = elapsed_seconds * 1000") {
        CppNet::Utils::Timer t("", false);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        double s = t.elapsed_seconds();
        double ms = t.elapsed_ms();
        ASSERT_NEAR(ms, s * 1000.0, 5.0); // 5ms tolerance
    } END_TEST("elapsed_ms = elapsed_seconds * 1000");

    TEST("Reset restarts timer") {
        CppNet::Utils::Timer t("", false);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        t.reset();
        double ms = t.elapsed_ms();
        ASSERT_TRUE(ms < 15.0); // should be much less than 20ms
    } END_TEST("Reset restarts timer");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
