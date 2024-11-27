#pragma once

#include <chrono>
#include <cstddef>

const size_t STEPS = 1024UL * 1024 * 1024 * 32;
const double dx = 1.0 / STEPS;

double func(double x) {
    return 4 / (1 + (x * x));
}

typedef std::chrono::steady_clock timer;
double get_time_ms(timer::duration dur) {
    return std::chrono::duration<double>(dur).count();
}