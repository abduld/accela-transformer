#include "utils.hpp"
#include "config.hpp"

static void Softmax_CPP_Naive(benchmark::State& state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(N,1), out(N);
  const auto inData = in.data();
  auto outData = out.data();
  for (auto _ : state) {
    const float alpha = *std::max_element(in.begin(), in.end());
    float denominator(0);
    for (int j = 0; j < N; j++) {
      outData[j] = std::exp(inData[j] - alpha);
      denominator += outData[j];
    }
    for (int j = 0; j < N; j++) {
      outData[j] /= denominator;
    }
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N*out[0];  // Expected to be 1
}

ADD_BENCHMARK(Softmax_CPP_Naive);