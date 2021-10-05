
#include <algorithm>
#include <cmath>
#include <vector>

#include <benchmark/benchmark.h>
#include <xsimd/xsimd.hpp>

#include "config.hpp"

namespace xs = xsimd;

#pragma once

static void Softmax_CPP_SIMD_OpenMP(benchmark::State& state) {
  using simd_t = xsimd::simd_type<float>;
  std::vector<float> in(N, 1), out(N, 1);

  auto inc = simd_t::size;
  // size for which the vectorization is possible
  constexpr auto vec_size = N - (N % inc);

  float* inData  = reinterpret_cast<float*>(__builtin_assume_aligned(in.data(), 64));
  float* outData = reinterpret_cast<float*>(__builtin_assume_aligned(out.data(), 64));

  float lastVal = 0;
  for (auto _ : state) {
    float maxVal = xsimdMax(inData, N);
    for (int ii = 0; ii < vec_size; ii += inc) {
      auto x = xsimd::exp(xsimd::load_aligned(&inData[ii]) - maxVal);
      x.store_aligned(&outData[ii]);
    }
    for (int ii = vec_size; ii < N; ii++) {
      outData[ii] = std::expf(inData[ii] - maxVal);
    }
    float totalVal = xsimdTotal(outData, N);
    for (int ii = 0; ii < vec_size; ii += inc) {
      auto x = xsimd::load_aligned(&outData[ii]) / totalVal;
      x.store_aligned(&outData[ii]);
    }
    for (int ii = vec_size; ii < N; ii++) {
      outData[ii] /= totalVal;
    }
    benchmark::ClobberMemory();
  }

  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
}
