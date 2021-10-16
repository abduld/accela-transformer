
#include "config.hpp"
#include "utils.hpp"

#include "naive.hat"

static void BENCHMARK_NAME(CPP_Naive)(benchmark::State& state) {

  aligned_vector<float> Input(N, 1.0/N);
  float output = 0;
  for (auto _ : state) {
    float accum = 0;
    const float * inputData = Input.data();
    for (int ii = 0 ; ii < N; ii++) {
      accum += *inputData++;
    }
    output = accum;
    benchmark::DoNotOptimize(Input.data());
    benchmark::ClobberMemory();
  }
  state.counters["Value"] = output;   
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive));

