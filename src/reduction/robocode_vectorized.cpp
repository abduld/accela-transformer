
#include "config.hpp"
#include "utils.hpp"

#include "vectorized.hat"

static constexpr int VectorSize = 8;

static void BENCHMARK_NAME(Robocode_Vectorized)(benchmark::State& state) {
  aligned_vector<float> Input(N, 1.0/N);
  float output = 0;
  for (auto _ : state) {
    output = 0;
    std::array<float, VectorSize> SumVec{0};
    vectorized(&output, Input.data(), SumVec.data());
    benchmark::DoNotOptimize(Input.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = output; // Expected to be 1 
}

ADD_BENCHMARK(BENCHMARK_NAME(Robocode_Vectorized));

