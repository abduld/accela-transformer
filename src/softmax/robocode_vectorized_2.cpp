#include "config.hpp"
#include "utils.hpp"

#include "vectorized_2.hat"

static void Robocode_Vectorized(benchmark::State& state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(N, 1), out(N);
  const auto inData = in.data();
  auto outData      = out.data();
  for (auto _ : state) {
    float denom = 0, maxVal = -std::numeric_limits<float>::max();
    vectorized_2_max(&maxVal, inData);
    vectorized_2_exp(outData, inData, &maxVal);
    vectorized_2_accum(&denom, outData);
    vectorized_2_div(&denom, outData);
    benchmark::DoNotOptimize(outData);
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(Robocode_Vectorized);
