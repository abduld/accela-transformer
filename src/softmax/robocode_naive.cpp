#include <vector>

#include "utils.hpp"
#include "config.hpp"

#include "naive.hat"


static void Robocode_Naive(benchmark::State& state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(N,1), out(N);
  const auto inData = in.data();
  auto outData = out.data();
  for (auto _ : state) {
    naive(outData, inData);
    benchmark::DoNotOptimize(outData);
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N*out[0];  // Expected to be 1
}

ADD_BENCHMARK(Robocode_Naive);
