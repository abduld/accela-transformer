#include "config.hpp"

static void BENCHMARK_NAME(CPP_SIMD_OpenMP_BatchFirst)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N, 1), out(BATCH_SIZE * N);
  for (auto _ : state) {
    /// [batch-first]
    for (int i = 0; i < BATCH_SIZE; i++) {
      const auto inData = in.data() + i * N;
      auto outData      = out.data() + i * N;
      auto maxVal       = std::numeric_limits<float>::min();
#pragma omp simd reduction(max : maxVal) aligned(inData : XSIMD_DEFAULT_ALIGNMENT)
      for (int j = 0; j < N; j++) {
        maxVal = std::max(maxVal, inData[j]);
      }
      float sum = 0;
#pragma omp simd reduction(+ : sum) aligned(inData, outData : XSIMD_DEFAULT_ALIGNMENT)
      for (int j = 0; j < N; j++) {
        outData[j] = std::exp(inData[j] - maxVal);
        sum += outData[j];
      }
#pragma omp simd aligned(outData : XSIMD_DEFAULT_ALIGNMENT)
      for (int j = 0; j < N; j++) {
        outData[j] /= sum;
      }
    }
    /// [batch-first]
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
  aligned_vector<float> in(BATCH_SIZE * N, 1), out(BATCH_SIZE * N);
  for (auto _ : state) {
    /// [length-first]
    aligned_vector<float> maxElements(BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int i = 0; i < BATCH_SIZE; i++) {
        const auto inData = in.data() + i * N;
        maxElements[i]    = std::max(maxElements[i], inData[jj]);
      }
    }

    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int i = 0; i < BATCH_SIZE; i++) {
        const auto inData = in.data() + i * N;
        auto outData      = out.data() + i * N;
        outData[jj]       = std::exp(inData[jj] - maxElements[i]);
        denominator[i] += outData[jj];
      }
    }

    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int i = 0; i < BATCH_SIZE; i++) {
        auto outData = out.data() + i * N;
        outData[jj] /= denominator[i];
      }
    }
    /// [length-first]

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
  aligned_vector<float> in(BATCH_SIZE * N, 1), out(BATCH_SIZE * N);
  for (auto _ : state) {
    /// [mixed]
    aligned_vector<float> maxElements(BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int i = 0; i < BATCH_SIZE; i++) {
        const auto inData = in.data() + i * N;
        maxElements[i]    = std::max(maxElements[i], inData[jj]);
      }
    }

    for (int jj = 0; jj < N; jj++) {
#pragma omp simd
      for (int i = 0; i < BATCH_SIZE; i++) {
        const auto inData = in.data() + i * N;
        auto outData      = out.data() + i * N;
        outData[jj]       = std::exp(inData[jj] - maxElements[i]);
        denominator[i] += outData[jj];
      }
    }

    for (int i = 0; i < BATCH_SIZE; i++) {
      auto outData     = out.data() + i * N;
      const auto denom = denominator[i];
#pragma omp simd
      for (int jj = 0; jj < N; jj++) {
        outData[jj] /= denom;
      }
    }
    /// [mixed]

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