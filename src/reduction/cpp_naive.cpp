
#include "config.hpp"
#include "utils.hpp"

static void BENCHMARK_NAME(CPP_Naive)(benchmark::State& state) {
  aligned_vector<float> Input(N, 1.0/N);
  float output = 0;
  for (auto _ : state) {
    output = std::accumulate(Input.begin(), Input.end(), 0.0f);
    benchmark::DoNotOptimize(Input.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = output; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive));

