#include "config.hpp"

/// [import-hat]
#include "vectorized.hat"
/// [import-hat]

static void BENCHMARK_NAME(Robocode_Vectorized)(benchmark::State& state) {
/// [declare-io]
  aligned_vector<float> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  const auto inData = in.data();
  auto outData      = out.data();
/// [declare-io]
  for (auto _ : state) {
/// [use-function]
    aligned_vector<float> maxElements(
        BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    auto maxData = maxElements.data(), denomData = denominator.data(); 
    vectorized(outData, inData, maxData, denomData); 
/// [use-function]
    benchmark::DoNotOptimize(outData);
    benchmark::DoNotOptimize(maxData);
    benchmark::DoNotOptimize(denomData);
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(Robocode_Vectorized));
