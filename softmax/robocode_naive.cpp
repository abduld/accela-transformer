#include <vector>
#include <benchmark/benchmark.h>

#include "softmax_naive.hat"


static void Softmax_RobocodeNaive(benchmark::State& state) {
  std::vector<float> in(N, 1), out(N, 1);
  for (auto _ : state) {
    softmax_naive(out.data(), in.data());
  }
  state.SetItemsProcessed(N);
}

BENCHMARK(Softmax_RobocodeNaive);
