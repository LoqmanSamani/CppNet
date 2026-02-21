/**
 * @file elapsed_time.cpp
 * @brief Implementation of Timer utility for measuring execution time
 */

#include "CppNet/utils/elapsed_time.hpp"

namespace CppNet
{
    namespace Utils
    {
        Timer::Timer(const std::string& label, bool verbose)
            : label_(label), verbose_(verbose),
              start_(std::chrono::high_resolution_clock::now())
        {
        }

        Timer::~Timer()
        {
            if (verbose_)
            {
                double ms = elapsed_ms();
                if (!label_.empty())
                    std::cout << "[" << label_ << "] ";
                std::cout << "Elapsed: " << ms << " ms ("
                          << ms / 1000.0 << " s)" << std::endl;
            }
        }

        double Timer::elapsed_seconds() const
        {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double>(now - start_).count();
        }

        double Timer::elapsed_ms() const
        {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(now - start_).count();
        }

        void Timer::reset()
        {
            start_ = std::chrono::high_resolution_clock::now();
        }
    }
}