
#include "config.hpp"
#include "utils.hpp"


namespace xs = xsimd;

static void BENCHMARK_NAME(CPP_XSIMD)(benchmark::State &state) {
  using simd_t = xsimd::simd_type<float>;
  aligned_vector<float> in(N, 1), out(N);

  float *inData =
      reinterpret_cast<float *>(__builtin_assume_aligned(in.data(), XSIMD_DEFAULT_ALIGNMENT));
  float *outData =
      reinterpret_cast<float *>(__builtin_assume_aligned(out.data(), XSIMD_DEFAULT_ALIGNMENT));

  for (auto _ : state) {
/// [max-val]
    float maxVal = xsimd::reduce(inData, inData + N, inData[0],
                                 [=](const auto &x, const auto &y) { return xsimd::max(x, y); });
                                 
/// [max-val]
/// [sum-exp]
    xsimd::transform(inData, inData + N, outData,
                     [=](const auto &x) { return xsimd::exp(x - maxVal); });
    float totalVal = xsimd::reduce(outData, outData + N, 0.0f);
/// [sum-exp]
/// [divide]
    xsimd::transform(outData, outData + N, outData, [=](const auto &x) { return x / totalVal; });
/// [divide]
    benchmark::DoNotOptimize(outData);
    benchmark::ClobberMemory();
  }

  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_XSIMD));