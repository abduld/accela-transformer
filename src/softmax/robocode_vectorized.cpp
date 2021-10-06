#include "utils.hpp"
#include "config.hpp"

#include "vectorized.hat"


static void Robocode_Vectorized(benchmark::State& state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(N,1), out(N);
  for (auto _ : state) {
    vectorized(out.data(), in.data());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N*out[0];  // Expected to be 1
}

ADD_BENCHMARK(Robocode_Vectorized);
