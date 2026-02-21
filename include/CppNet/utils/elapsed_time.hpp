/**
 * @file elapsed_time.hpp
 * @brief Timer utilities for benchmarking CppNet operations
 */

#ifndef ELAPSED_TIME_HPP
#define ELAPSED_TIME_HPP

#include <chrono>
#include <string>
#include <iostream>

namespace CppNet
{
    namespace Utils
    {
        /**
         * @class Timer
         * @brief A simple RAII timer for measuring elapsed time.
         *
         * @example
         *   {
         *       CppNet::Utils::Timer t("Forward pass");
         *       model.forward(input);
         *   } // prints elapsed time on destruction
         */
        class Timer
        {
        public:
            explicit Timer(const std::string& label = "", bool verbose = true);
            ~Timer();

            double elapsed_seconds() const;
            double elapsed_ms() const;
            void reset();

        private:
            std::string label_;
            bool verbose_;
            std::chrono::high_resolution_clock::time_point start_;
        };
    }
}

#endif // ELAPSED_TIME_HPP