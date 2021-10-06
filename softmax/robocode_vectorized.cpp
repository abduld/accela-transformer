#include <vector>

#include "utils.hpp"
#include "config.hpp"

#include "softmax_vectorized.hat"


static void Softmax_RobocodeVectorized(benchmark::State& state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(N,1), out(N);
  for (auto _ : state) {
    softmax_vectorized(out.data(), in.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N*out[0];  // Expected to be 1
}

ADD_BENCHMARK(Softmax_RobocodeVectorized);
