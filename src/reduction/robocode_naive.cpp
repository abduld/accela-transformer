
#include "config.hpp"
#include "utils.hpp"

#include "naive.hat"

static void BENCHMARK_NAME(Robocode_Naive)(benchmark::State& state) {
  aligned_vector<float> Input(N, 1.0/N);
  float output = 0;
  for (auto _ : state) {
    output = 0;
    naive(&output, Input.data());
    benchmark::DoNotOptimize(Input.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = output; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(Robocode_Naive));

