
#include "config.hpp"
#include "utils.hpp"

static void BENCHMARK_NAME(CPP_SIMD_OpenMP)(benchmark::State &state) {
  aligned_vector<float> in(N, 1), out(N);
  const auto inData = in.data();
  auto outData      = out.data();
  for (auto _ : state) {
/// [max-val]
    auto maxVal = -std::numeric_limits<float>::min();
#pragma omp simd reduction(max : maxVal) aligned(inData : XSIMD_DEFAULT_ALIGNMENT)
    for (int idx = 0; idx < N; idx++) {
      maxVal = std::max(maxVal, inData[idx]);
    }
/// [max-val]
/// [sum-exp]
    float sum = 0;
#pragma omp simd reduction(+ : sum) aligned(inData, outData : XSIMD_DEFAULT_ALIGNMENT)
    for (int idx = 0; idx < N; idx++) {
      outData[idx] = std::exp(inData[idx] - maxVal);
      sum += outData[idx];
    }
/// [sum-exp]
/// [divide]
#pragma omp simd aligned(outData : XSIMD_DEFAULT_ALIGNMENT)
    for (int idx = 0; idx < N; idx++) {
      outData[idx] /= sum;
    }
/// [divide]
    benchmark::DoNotOptimize(outData);
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_SIMD_OpenMP));