/**
 * @file test_visualizations.cpp
 * @brief Unit tests for TrainingLogger
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <fstream>
#include "CppNet/visualizations/visualizations.hpp"

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
    std::cout << "=== TrainingLogger Tests ===\n";

    TEST("Construction") {
        CppNet::Visualizations::TrainingLogger logger;
        ASSERT_TRUE(logger.current_epoch() == 0);
    } END_TEST("Construction");

    TEST("Log and retrieve") {
        CppNet::Visualizations::TrainingLogger logger;
        logger.log("loss", 0.5f);
        auto history = logger.get_history("loss");
        ASSERT_TRUE(history.size() == 1);
        ASSERT_NEAR(history[0].second, 0.5f, 1e-6f);
    } END_TEST("Log and retrieve");

    TEST("next_epoch advances counter") {
        CppNet::Visualizations::TrainingLogger logger;
        logger.log("loss", 1.0f);
        logger.next_epoch();
        ASSERT_TRUE(logger.current_epoch() == 1);
        logger.log("loss", 0.5f);
        auto history = logger.get_history("loss");
        ASSERT_TRUE(history.size() == 2);
        ASSERT_TRUE(history[0].first == 0);
        ASSERT_TRUE(history[1].first == 1);
    } END_TEST("next_epoch advances counter");

    TEST("Multiple metrics") {
        CppNet::Visualizations::TrainingLogger logger;
        logger.log("loss", 1.0f);
        logger.log("accuracy", 0.8f);
        auto names = logger.get_metric_names();
        ASSERT_TRUE(names.size() == 2);
    } END_TEST("Multiple metrics");

    TEST("get_metric_names returns all") {
        CppNet::Visualizations::TrainingLogger logger;
        logger.log("train_loss", 1.0f);
        logger.log("val_loss", 0.9f);
        logger.log("train_acc", 0.7f);
        auto names = logger.get_metric_names();
        ASSERT_TRUE(names.size() == 3);
    } END_TEST("get_metric_names returns all");

    TEST("export_csv creates file") {
        CppNet::Visualizations::TrainingLogger logger;
        logger.log("loss", 0.5f);
        logger.next_epoch();
        logger.log("loss", 0.3f);
        const char* path = "/tmp/cppnet_test_logger.csv";
        bool ok = logger.export_csv(path);
        ASSERT_TRUE(ok);
        // Verify file exists and has content
        std::ifstream f(path);
        ASSERT_TRUE(f.good());
        std::string line;
        std::getline(f, line);
        ASSERT_TRUE(!line.empty());
        f.close();
        std::remove(path);
    } END_TEST("export_csv creates file");

    TEST("clear resets everything") {
        CppNet::Visualizations::TrainingLogger logger;
        logger.log("loss", 1.0f);
        logger.next_epoch();
        logger.log("loss", 0.5f);
        logger.clear();
        ASSERT_TRUE(logger.current_epoch() == 0);
        auto names = logger.get_metric_names();
        ASSERT_TRUE(names.size() == 0);
    } END_TEST("clear resets everything");

    TEST("Empty history for unknown metric") {
        CppNet::Visualizations::TrainingLogger logger;
        auto history = logger.get_history("nonexistent");
        ASSERT_TRUE(history.size() == 0);
    } END_TEST("Empty history for unknown metric");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
