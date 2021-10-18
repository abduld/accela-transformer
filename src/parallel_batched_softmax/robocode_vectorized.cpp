#include "config.hpp"

#include "vectorized.hat"

static void BENCHMARK_NAME(Accera_Vectorized)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  const auto inData = in.data();
  auto outData      = out.data();
  for (auto _ : state) {
    aligned_vector<float> maxElements(
        BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    auto maxData = maxElements.data(), denomData = denominator.data(); 
    vectorized(outData, inData, maxData, denomData); 
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

ADD_BENCHMARK(BENCHMARK_NAME(Accera_Vectorized));
