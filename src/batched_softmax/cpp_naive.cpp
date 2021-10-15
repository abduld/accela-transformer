#include "config.hpp"
#include "utils.hpp"

static void BENCHMARK_NAME(CPP_Naive_BatchFirst)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N,
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

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive_BatchFirst));

static void BENCHMARK_NAME(CPP_Naive_LengthFirst)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  for (auto _ : state) {
    aligned_vector<float> maxElements(BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    for (int jj = 0; jj < N; jj++) {
      for (int ii = 0; ii < BATCH_SIZE; ii++) {
        const auto inData = in.data() + ii * N;
        maxElements[ii]   = std::max(maxElements[ii], inData[jj]);
      }
    }

    for (int jj = 0; jj < N; jj++) {
      for (int ii = 0; ii < BATCH_SIZE; ii++) {
        const auto inData = in.data() + ii * N;
        auto outData      = out.data() + ii * N;
        outData[jj]       = std::exp(inData[jj] - maxElements[ii]);
        denominator[ii] += outData[jj];
      }
    }

    for (int jj = 0; jj < N; jj++) {
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

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive_LengthFirst));

static void BENCHMARK_NAME(CPP_Naive_Mixed)(benchmark::State& state) {
  aligned_vector<float> in(BATCH_SIZE * N,
                                                                                  1),
      out(BATCH_SIZE * N);
  for (auto _ : state) {
    aligned_vector<float> maxElements(BATCH_SIZE, std::numeric_limits<float>::min()),
        denominator(BATCH_SIZE, 0);
    for (int jj = 0; jj < N; jj++) {
      for (int ii = 0; ii < BATCH_SIZE; ii++) {
        const auto inData = in.data() + ii * N;
        maxElements[ii]   = std::max(maxElements[ii], inData[jj]);
      }
    }

    for (int jj = 0; jj < N; jj++) {
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

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive_Mixed));