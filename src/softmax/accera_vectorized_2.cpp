#include "config.hpp"

/// [import-hat]
#include "vectorized_2.hat"
/// [import-hat]

static void BENCHMARK_NAME(Accera_Vectorized_2)(benchmark::State& state) {
/// [declare-io]
  aligned_vector<float> in(N, 1), out(N);
  const auto inData = in.data();
  auto outData      = out.data();
/// [declare-io]
  for (auto _ : state) {
/// [use-function]
    float denom = 0, maxVal = std::numeric_limits<float>::min();
    vectorized_2_max(&maxVal, inData);
    vectorized_2_exp(outData, inData, &maxVal);
    vectorized_2_accum(&denom, outData);
    vectorized_2_div(&denom, outData);
/// [use-function]
    benchmark::DoNotOptimize(outData);
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(Accera_Vectorized_2));
