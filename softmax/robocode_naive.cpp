#include <vector>
#include <benchmark/benchmark.h>

#include "config.hpp"

#include "softmax_naive.hat"


static void Softmax_RobocodeNaive(benchmark::State& state) {
  std::vector<float> in(N, 1), out(N, 0);
  for (auto _ : state) {
    softmax_naive(out.data(), in.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
}

BENCHMARK(Softmax_RobocodeNaive)->Unit(benchmark::kMillisecond);
