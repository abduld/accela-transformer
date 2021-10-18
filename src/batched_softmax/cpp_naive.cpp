#include "config.hpp"

static void BENCHMARK_NAME(CPP_Naive_BatchFirst)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N, 1), out(BATCH_SIZE * N);
  for (auto _ : state) {
    /// [batch-first]
    for (int i = 0; i < BATCH_SIZE; i++) {
      const auto inStart = in.begin() + i * N;
      const auto inEnd   = in.begin() + (i + 1) * N;
      const auto inData  = in.data() + i * N;
      auto outData       = out.data() + i * N;
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
    /// [batch-first]
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const int64_t items_processed = state.iterations() * N * BATCH_SIZE;
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(float));
  state.counters["Value"] = N * out[0]; // Expected to be 1
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive_BatchFirst));

static void BENCHMARK_NAME(CPP_Naive_LengthFirst)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N, 1), out(BATCH_SIZE * N);
  for (auto _ : state) {
    /// [length-first]
    aligned_vector<float> maxElements(BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    for (int jj = 0; jj < N; jj++) {
      for (int i = 0; i < BATCH_SIZE; i++) {
        const auto inData = in.data() + i * N;
        maxElements[i]    = std::max(maxElements[i], inData[jj]);
      }
    }

    for (int jj = 0; jj < N; jj++) {
      for (int i = 0; i < BATCH_SIZE; i++) {
        const auto inData = in.data() + i * N;
        auto outData      = out.data() + i * N;
        outData[jj]       = std::exp(inData[jj] - maxElements[i]);
        denominator[i] += outData[jj];
      }
    }

    for (int jj = 0; jj < N; jj++) {
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

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive_LengthFirst));

static void BENCHMARK_NAME(CPP_Naive_Mixed)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N, 1), out(BATCH_SIZE * N);
  for (auto _ : state) {
    /// [mixed]
    aligned_vector<float> maxElements(BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    for (int jj = 0; jj < N; jj++) {
      for (int i = 0; i < BATCH_SIZE; i++) {
        const auto inData = in.data() + i * N;
        maxElements[i]    = std::max(maxElements[i], inData[jj]);
      }
    }

    for (int jj = 0; jj < N; jj++) {
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

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive_Mixed));