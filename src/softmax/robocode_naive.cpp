#include "config.hpp"
#include "utils.hpp"

/// [import-hat]
#include "naive.hat"
/// [import-hat]

static void BENCHMARK_NAME(Robocode_Naive)(benchmark::State& state) {
/// [declare-input]
  aligned_vector<float> in(N, 1), out(N);
  const auto inData = in.data();
  auto outData      = out.data();
/// [declare-input]
  for (auto _ : state) {
/// [use-function]
    naive(outData, inData);
/// [use-function]
    benchmark::DoNotOptimize(outData);
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(Robocode_Naive));
