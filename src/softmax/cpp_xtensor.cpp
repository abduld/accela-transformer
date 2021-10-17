
#include "config.hpp"
#include "utils.hpp"

#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>

static void BENCHMARK_NAME(CPP_XTensor)(benchmark::State &state) {
  aligned_vector<float> in(N, 1), out(N);
  auto inTensor = xt::adapt(in, {N}), outTensor = xt::adapt(out, {N});

  for (auto _ : state) {
/// [max-val]
    float maxVal           = xt::amax(inTensor)[0];
/// [max-val]
/// [sum-exp]
    xt::noalias(outTensor) = xt::exp(inTensor - maxVal);
    float totalVal         = xt::sum(outTensor)[0];
/// [sum-exp]
/// [divide]
    xt::noalias(outTensor) /= totalVal;
/// [divide]
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_XTensor));