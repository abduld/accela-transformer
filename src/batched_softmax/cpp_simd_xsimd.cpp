#include "config.hpp"
#include "utils.hpp"

static void CPP_XSIMD(benchmark::State &state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  for (auto _ : state) {
    for (int ii = 0; ii < BATCH_SIZE; ii++) {
      const auto inData  = in.data() + ii * N;
      auto outData       = out.data() + ii * N;
      float maxVal       = xsimd::reduce(inData, inData + N, inData[0],
                                   [=](const auto &x, const auto &y) { return xsimd::max(x, y); });
      xsimd::transform(inData, inData + N, outData,
                       [=](const auto &x) { return xsimd::exp(x - maxVal); });
      float totalVal = xsimd::reduce(outData, outData + N, 0.0f);
      xsimd::transform(outData, outData + N, outData, [=](const auto &x) { return x / totalVal; });
    }
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(CPP_XSIMD);