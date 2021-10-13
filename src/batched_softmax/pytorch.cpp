// use softmax function as defined in
// https://pytorch.org/cppdocs/api/function_namespaceat_1aa0a610d0d2cafaa3335bc5122420f2fb.html#exhale-function-namespaceat-1aa0a610d0d2cafaa3335bc5122420f2fb
// which should eventually call
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SoftMax.cpp#L141

#include <torch/torch.h>

#include "config.hpp"
#include "utils.hpp"

static void BENCHMARK_NAME(Pytorch)(benchmark::State& state) {
  at::init_num_threads();
  at::set_num_threads(1);
  auto options      = at::TensorOptions().dtype(torch::kFloat32).device(at::kCPU).requires_grad(false);
  torch::Tensor in  = torch::ones({BATCH_SIZE, N}, options),
                out = torch::zeros({BATCH_SIZE, N}, options);
  for (auto _ : state) {
    out.set_(at::_softmax(in, 0, false));
    benchmark::DoNotOptimize(in.data_ptr());
    benchmark::DoNotOptimize(out.data_ptr());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out.data_ptr<float>()[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(Pytorch));