#include "config.hpp"

static void BENCHMARK_NAME(CPP_SIMD_OpenMP_BatchFirst)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  for (auto _ : state) {
/// [algorithm]
    for (int ii = 0; ii < BATCH_SIZE; ii++) {
      const auto inData = in.data() + ii * N;
      auto outData      = out.data() + ii * N;
      auto maxVal       = std::numeric_limits<float>::min();
#pragma omp simd reduction(max : maxVal) aligned(inData : XSIMD_DEFAULT_ALIGNMENT)
      for (int idx = 0; idx < N; idx++) {
        maxVal = std::max(maxVal, inData[idx]);
      }
      float sum = 0;
#pragma omp simd reduction(+ : sum) aligned(inData, outData : XSIMD_DEFAULT_ALIGNMENT)
      for (int idx = 0; idx < N; idx++) {
        outData[idx] = std::exp(inData[idx] - maxVal);
        sum += outData[idx];
      }
#pragma omp simd aligned(outData : XSIMD_DEFAULT_ALIGNMENT)
      for (int idx = 0; idx < N; idx++) {
        outData[idx] /= sum;
      }
    }
/// [algorithm]
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_SIMD_OpenMP_BatchFirst));

static void BENCHMARK_NAME(CPP_SIMD_OpenMP_LengthFirst)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  for (auto _ : state) {
    aligned_vector<float> maxElements(
        BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int ii = 0; ii < BATCH_SIZE; ii++) {
        const auto inData = in.data() + ii * N;
        maxElements[ii]   = std::max(maxElements[ii], inData[jj]);
      }
    }

    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int ii = 0; ii < BATCH_SIZE; ii++) {
        const auto inData = in.data() + ii * N;
        auto outData      = out.data() + ii * N;
        outData[jj]       = std::exp(inData[jj] - maxElements[ii]);
        denominator[ii] += outData[jj];
      }
    }

    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int ii = 0; ii < BATCH_SIZE; ii++) {
        auto outData = out.data() + ii * N;
        outData[jj] /= denominator[ii];
      }
    }

    benchmark::DoNotOptimize(out.data());
    benchmark::DoNotOptimize(maxElements.data());
    benchmark::DoNotOptimize(denominator.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_SIMD_OpenMP_LengthFirst));

static void BENCHMARK_NAME(CPP_SIMD_OpenMP_Mixed)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  for (auto _ : state) {
    aligned_vector<float> maxElements(
        BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int ii = 0; ii < BATCH_SIZE; ii++) {
        const auto inData = in.data() + ii * N;
        maxElements[ii]   = std::max(maxElements[ii], inData[jj]);
      }
    }

    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int ii = 0; ii < BATCH_SIZE; ii++) {
        const auto inData = in.data() + ii * N;
        auto outData      = out.data() + ii * N;
        outData[jj]       = std::exp(inData[jj] - maxElements[ii]);
        denominator[ii] += outData[jj];
      }
    }

    for (int ii = 0; ii < BATCH_SIZE; ii++) {
      auto outData     = out.data() + ii * N;
      const auto denom = denominator[ii];
#pragma omp simd
      for (int jj = 0; jj < N; jj++) {
        outData[jj] /= denom;
      }
    }

    benchmark::DoNotOptimize(out.data());
    benchmark::DoNotOptimize(maxElements.data());
    benchmark::DoNotOptimize(denominator.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_SIMD_OpenMP_Mixed));