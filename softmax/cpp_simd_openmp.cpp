#include <algorithm>
#include <cmath>
#include <vector>

#include <benchmark/benchmark.h>
#include <xsimd/xsimd.hpp> 

#include "config.hpp"

static void Softmax_CPP_SIMD_OpenMP(benchmark::State& state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(N, 1), out(N, 0);
  const auto inData = in.data();
  auto outData      = out.data();
  for (auto _ : state) {
    auto maxVal = -std::numeric_limits<float>::max();
#pragma omp simd reduction(max : maxVal) aligned(inData : 16)
    for (int idx = 0; idx < N; idx++) {
      maxVal = std::max(maxVal, inData[idx]);
    }
    float sum = 0;
#pragma omp simd reduction(+ : sum) aligned(inData, outData : 16)
    for (int idx = 0; idx < N; idx++) {
      outData[idx] = expf(inData[idx] - maxVal);
      sum += outData[idx];
    }
#pragma omp simd aligned(outData : 32)
    for (int idx = 0; idx < N; idx++) {
      outData[idx] /= sum;
    }
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
}

BENCHMARK(Softmax_CPP_SIMD_OpenMP)->Unit(benchmark::kMillisecond);