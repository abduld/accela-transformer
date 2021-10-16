
#include "config.hpp"
#include "utils.hpp"

#include "vectorized.hat"

static void BENCHMARK_NAME(Robocode_Vectorized)(benchmark::State& state) {

  aligned_vector<float> Input(N, 1.0/N);
  float output = 0;
  for (auto _ : state) {
    vectorized(&output, Input.data());
    benchmark::DoNotOptimize(Input.data());
    benchmark::ClobberMemory();
  }
  state.counters["Value"] = output;   
}

ADD_BENCHMARK(BENCHMARK_NAME(Robocode_Vectorized));

