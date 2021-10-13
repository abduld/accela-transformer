// use softmax function as defined in
// https://pytorch.org/cppdocs/api/function_namespaceat_1aa0a610d0d2cafaa3335bc5122420f2fb.html#exhale-function-namespaceat-1aa0a610d0d2cafaa3335bc5122420f2fb
// which should eventually call
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SoftMax.cpp#L141

#include <torch/torch.h>

#include "config.hpp"
#include "utils.hpp"

static void BENCHMARK_NAME(Pytorch)(benchmark::State& state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  for (auto _ : state) {
    torch::Tensor tensor = torch::eye(3);
    benchmark::DoNotOptimize(tensor.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(Pytorch));

BENCHMARK_MAIN();