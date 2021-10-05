#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <benchmark/benchmark.h>

#include "config.hpp"

static void Softmax_CPP_Naive(benchmark::State& state) {
  std::vector<float> in(N, 1), out(N, 1);
  const auto inData = in.data();
  auto outData = out.data();
  for (auto _ : state) {
    const float alpha = *std::max_element(in.begin(), in.end());
    float denominator(0);
    for (int j = 0; j < N; j++) {
      outData[j] = std::exp(inData[j] - alpha);
      denominator += outData[j];
    }
    for (int j = 0; j < N; j++) {
      outData[j] /= denominator;
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(N);
  state.SetBytesProcessed(N * sizeof(float));
}
