#include "config.hpp"

static void BENCHMARK_NAME(CPP_Naive)(benchmark::State& state) {
  aligned_vector<float> in(N, 1), out(N);
  const auto inData = in.data();
  auto outData      = out.data();
  for (auto _ : state) {
    /// [max-val]
    const float max = *std::max_element(in.begin(), in.end());
    /// [max-val]
    /// [sum-exp]
    float denominator(0);
    for (int j = 0; j < N; j++) {
      outData[j] = std::exp(inData[j] - max);
      denominator += outData[j];
    }
    /// [sum-exp]
    /// [divide]
    for (int j = 0; j < N; j++) {
      outData[j] /= denominator;
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

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive));