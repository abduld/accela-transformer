
#include "utils.hpp"
#include "config.hpp"

namespace xs = xsimd;


static float xsimdMax(const float* data0, int len) {
  const float* data = reinterpret_cast<float*>(__builtin_assume_aligned(data0, XSIMD_DEFAULT_ALIGNMENT));
  float maxVal      = data[0];
#pragma omp simd reduction(max : maxVal)
  for (int ii = 0; ii < len; ii++) {
    maxVal = std::max(maxVal, data[ii]);
  }
  return maxVal;
}


static float xsimdTotal(const float* data0, int len) {
  using simd_t      = xsimd::simd_type<float>;
  const float* data = reinterpret_cast<float*>(__builtin_assume_aligned(data0, XSIMD_DEFAULT_ALIGNMENT));

  auto inc     = simd_t::size;
  auto vecSize = len - len % inc;

  simd_t stotalVal = xsimd::load_aligned(&data[0]);
  for (int ii = inc; ii < vecSize; ii += inc) {
    stotalVal += xsimd::load_aligned(&data[ii]);
  }
  float totalVal = xsimd::hadd(stotalVal);
  for (int ii = vecSize; ii < len; ii++) {
    totalVal += data[ii];
  }
  return totalVal;
}

static void CPP_XSIMD(benchmark::State& state) {
  using simd_t      = xsimd::simd_type<float>;
  std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> in(N,1), out(N);

  constexpr int inc = simd_t::size;
  // size for which the vectorization is possible
  constexpr auto vec_size = N - (N % inc);

  float* inData  = reinterpret_cast<float*>(__builtin_assume_aligned(in.data(), XSIMD_DEFAULT_ALIGNMENT));
  float* outData = reinterpret_cast<float*>(__builtin_assume_aligned(out.data(), XSIMD_DEFAULT_ALIGNMENT));

  for (auto _ : state) {
    float maxVal = xsimdMax(inData, N);
    for (int ii = 0; ii < vec_size; ii += inc) {
      auto x = xsimd::exp(xsimd::load_aligned(&inData[ii]) - maxVal);
      x.store_aligned(&outData[ii]);
    }
    for (int ii = vec_size; ii < N; ii++) {
      outData[ii] = std::exp(inData[ii] - maxVal);
    }
    float totalVal = xsimdTotal(outData, N);
    for (int ii = 0; ii < vec_size; ii += inc) {
      auto x = xsimd::load_aligned(&outData[ii]) / totalVal;
      x.store_aligned(&outData[ii]);
    }
    for (int ii = vec_size; ii < N; ii++) {
      outData[ii] /= totalVal;
    }
    benchmark::ClobberMemory();
  }

  const int64_t items_processed = state.iterations() * N;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N*out[0];  // Expected to be 1
}

ADD_BENCHMARK(CPP_XSIMD);