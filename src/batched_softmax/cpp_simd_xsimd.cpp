#include "config.hpp"
#include "utils.hpp"

static void BENCHMARK_NAME(CPP_XSIMD_BatchFirst)(benchmark::State &state) {
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  for (auto _ : state) {
    for (int ii = 0; ii < BATCH_SIZE; ii++) {
      const auto inData = in.data() + ii * N;
      auto outData      = out.data() + ii * N;
      float maxVal      = xsimd::reduce(inData, inData + N, inData[0],
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

ADD_BENCHMARK(BENCHMARK_NAME(CPP_XSIMD_BatchFirst));


// static void BENCHMARK_NAME(CPP_XSIMD_LengthFirst)(benchmark::State &state) {
//   std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(BATCH_SIZE * N,
//                                                                                   1),
//       out(BATCH_SIZE * N);
//   for (auto _ : state) {

//     using simd_t           = xsimd::simd_type<float>;
//     constexpr auto inc     = simd_t::size;
//     constexpr auto vecSize = N - N % inc;
//     std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> maxElements(BATCH_SIZE, std::numeric_limits<float>::min()),
//         denominator(BATCH_SIZE, 0);
    
    
//     std::vector<float> maxElements(BATCH_SIZE, std::numeric_limits<float>::min()),
//         denominator(BATCH_SIZE, 0);
//     for (int jj = 0; jj < N; jj++) {
// #pragma omp simd
//       for (int ii = 0; ii < BATCH_SIZE; ii++) {
//         const auto inData = in.data() + ii * N;
//         maxElements[ii]   = std::max(maxElements[ii], inData[jj]);
//       }
//     }

//     for (int jj = 0; jj < N; jj++) {
// #pragma omp simd
//       for (int ii = 0; ii < BATCH_SIZE; ii++) {
//         const auto inData = in.data() + ii * N;
//         auto outData      = out.data() + ii * N;
//         outData[jj]       = std::exp(inData[jj] - maxElements[ii]);
//         denominator[ii] += outData[jj];
//       }
//     }

//     for (int jj = 0; jj < N; jj++) {
// #pragma omp simd
//       for (int ii = 0; ii < BATCH_SIZE; ii++) {
//         auto outData = out.data() + ii * N;
//         outData[jj] /= denominator[ii];
//       }
//     }

//     benchmark::DoNotOptimize(out.data());
//     benchmark::DoNotOptimize(maxElements.data());
//     benchmark::DoNotOptimize(denominator.data());
//     benchmark::ClobberMemory();
//   }
//   const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
//   state.SetItemsProcessed(items_processed);
//   state.SetBytesProcessed(items_processed * sizeof(float));
//   state.counters["Value"] = N * out[0]; // Expected to be 1
// }

// ADD_BENCHMARK(BENCHMARK_NAME(CPP_XSIMD_LengthFirst));