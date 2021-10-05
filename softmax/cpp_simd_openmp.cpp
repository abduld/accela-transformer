#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <benchmark/benchmark.h>

#include "config.hpp"

static void Softmax_CPP_SIMD_OpenMP(benchmark::State& state) {
  std::vector<float> in(N, 1), out(N, 1);
  const auto inData = in.data();
  auto outData      = out.data();
  for (auto _ : state) {
    auto maxVal = -std::numeric_limits<float>::max();
#pragma omp simd reduction(max : maxVal) aligned(currInput : 32)
    for (int idx = 0; idx < N; idx++) {
      maxVal = std::max(maxVal, inData[idx]);
    }
    float sum = 0;
#pragma omp simd reduction(+ : sum) aligned(currInput, currOutput : 32)
    for (int idx = 0; idx < N; idx++) {
      outData[idx] = expf(inData[idx] - maxVal);
      sum += outData[idx];
    }
#pragma omp simd aligned(currOutput : 32)
    for (int idx = 0; idx < N; idx++) {
      outData[idx] /= sum;
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(N);
  state.SetBytesProcessed(N * sizeof(float));
}
