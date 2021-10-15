#include "config.hpp"
#include "utils.hpp"

#include "vectorized.hat"

static void BENCHMARK_NAME(Robocode_Vectorized)(benchmark::State& state) {
  aligned_vector<float> in(N, 1), out(N);
  const auto inData = in.data();
  auto outData      = out.data();
  for (auto _ : state) {
    vectorized(outData, inData);
    benchmark::DoNotOptimize(outData);
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(Robocode_Vectorized));
