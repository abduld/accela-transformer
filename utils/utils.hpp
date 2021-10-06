#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include <benchmark/benchmark.h>
#include <xsimd/xsimd.hpp> 

#include "random.hpp"

#define ADD_BENCHMARK(fn) BENCHMARK(fn)->Unit(benchmark::kMillisecond)
