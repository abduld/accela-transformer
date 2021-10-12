#include "config.hpp"
#include "utils.hpp"

static void CPP_Naive(benchmark::State& state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  for (auto _ : state) {
    for (int ii = 0; ii < BATCH_SIZE; ii++) {
      const auto inStart = in.begin() + ii * N;
      const auto inEnd   = in.begin() + (ii + 1) * N;
      const auto inData  = in.data() + ii * N;
      auto outData       = out.data() + ii * N;
      const float max    = *std::max_element(inStart, inEnd);
      float denominator(0);
      for (int j = 0; j < N; j++) {
        outData[j] = std::exp(inData[j] - max);
        denominator += outData[j];
      }
      for (int j = 0; j < N; j++) {
        outData[j] /= denominator;
      }
    }
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(CPP_Naive);