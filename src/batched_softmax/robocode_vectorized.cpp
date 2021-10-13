#include "config.hpp"
#include "utils.hpp"

#include "vectorized.hat"

static void BENCHMARK_NAME(Robocode_Vectorized)(benchmark::State& state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  const auto inData = in.data();
  auto outData      = out.data();
  for (auto _ : state) {
#if 0
    std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> maxElements(
        BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    auto maxData = maxElements.data(), denomData = denominator.data();
    vectorized_max(maxData, inData);
    vectorized_exp(outData, inData, maxData);
    vectorized_accum(denomData, outData);
    vectorized_div(denomData, outData);
    benchmark::DoNotOptimize(maxData);
    benchmark::DoNotOptimize(denomData);
#else
    vectorized(outData, inData);
#endif
    benchmark::DoNotOptimize(outData);
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(Robocode_Vectorized));
