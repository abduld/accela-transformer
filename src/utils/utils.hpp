#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <numeric>
#include <stdint.h>
#include <string.h>

#include <benchmark/benchmark.h>
#include <xsimd/xsimd.hpp>

#include "random.hpp"

#ifndef MODULE_NAME
#error "MODULE_NAME is not defined"
#endif // MODULE_NAME

#define PP_STRINGIFY_IMPL(x) #x
#define PP_STRINGIFY(x) PP_STRINGIFY_IMPL(x)

#define PP_CONCAT_IMPL(x, y) x##y
#define PP_CONCAT(x, y) PP_CONCAT_IMPL(x, y)

#define BENCHMARK_NAME(fn_name) PP_CONCAT(MODULE_NAME, PP_CONCAT(_, fn_name))

#define ADD_BENCHMARK_(fn) BENCHMARK(fn)->Unit(benchmark::kMicrosecond)
#define ADD_BENCHMARK(fn) ADD_BENCHMARK_(fn)

template <typename T>
using aligned_vector =
    std::vector<T, xsimd::aligned_allocator<T, XSIMD_DEFAULT_ALIGNMENT>>;