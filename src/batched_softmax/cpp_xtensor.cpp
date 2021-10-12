#include "config.hpp"
#include "utils.hpp"

#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>

static void CPP_XTensor(benchmark::State &state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);

  std::vector<size_t> shape = {BATCH_SIZE, N};
  auto inTensor             = xt::adapt(in, shape);
  auto outTensor            = xt::adapt(out, shape);

  for (auto _ : state) {

    auto maxVal            = xt::amax(inTensor);
    xt::noalias(outTensor) = xt::exp(inTensor - maxVal);
    auto totalVal          = xt::sum(outTensor);
    xt::noalias(outTensor) /= totalVal;
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(CPP_XTensor);