#include <vector>

#include "config.hpp"
#include "utils.hpp"

#include "naive.hat"

static void BENCHMARK_NAME(Robocode_Naive)(benchmark::State& state) {

  aligned_vector<float> Q(SEQUENCE_LENGTH * DM, 1), K(SEQUENCE_LENGTH * DM, 1),
      V(SEQUENCE_LENGTH * DM, 1);
  aligned_vector<float> QK(SEQUENCE_LENGTH * SEQUENCE_LENGTH, -1), Output(SEQUENCE_LENGTH * DM, -1);

  for (auto _ : state) {
    aligned_vector<float> maxElements(SEQUENCE_LENGTH, std::numeric_limits<float>::min()),
        denominator(SEQUENCE_LENGTH, 0);
    naive_gemm_qk(Q.data(), K.data(), QK.data());
    naive_softmax(QK.data(), QK.data(), maxElements.data(), denominator.data());
    naive_gemm_qkv(QK.data(), V.data(), Output.data());
    benchmark::DoNotOptimize(Output.data());
    benchmark::DoNotOptimize(QK.data());
    benchmark::ClobberMemory();
  }
  state.counters["Value"] = Output[0];  
  state.counters["QK"] = QK[0];  
}

ADD_BENCHMARK(BENCHMARK_NAME(Robocode_Naive));
