
#include "config.hpp"
#include "utils.hpp"

#include "naive.hat"

static void BENCHMARK_NAME(Robocode_Naive)(benchmark::State& state) {

  aligned_vector<float> Input(N, 1.0/N);
  float output = 0;
  for (auto _ : state) {
    naive(&output, Input.data());
    benchmark::DoNotOptimize(Input.data());
    benchmark::ClobberMemory();
  }
  state.counters["Value"] = output;   
}

ADD_BENCHMARK(BENCHMARK_NAME(Robocode_Naive));

