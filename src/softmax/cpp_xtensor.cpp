
#include "config.hpp"
#include "utils.hpp"

#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>

static void CPP_XTensor(benchmark::State &state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(N, 1), out(N);
  auto inTensor = xt::adapt(in, {N}), outTensor = xt::adapt(out, {N});

  for (auto _ : state) {
    float maxVal           = xt::amax(inTensor)[0];
    xt::noalias(outTensor) = xt::exp(inTensor - maxVal);
    float totalVal         = xt::sum(outTensor)[0];
    xt::noalias(outTensor) /= totalVal;
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(CPP_XTensor);