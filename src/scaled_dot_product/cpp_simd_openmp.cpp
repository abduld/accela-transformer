#include "config.hpp"

#include <cblas.h>

/// [row-softmax]
template <int NRows, int NCols>
static void row_softmax(float *outData0, const float *inData0) {
  for (int i = 0; i < NRows; i++) {
    const auto inData = inData0 + i * N;
    auto outData      = outData0 + i * N;
    auto maxVal       = std::numeric_limits<float>::min();
#pragma omp simd reduction(max : maxVal) aligned(inData : XSIMD_DEFAULT_ALIGNMENT)
    for (int j = 0; j < NCols; j++) {
      maxVal = std::max(maxVal, inData[j]);
    }
    float sum = 0;
#pragma omp simd reduction(+ : sum) aligned(inData, outData : XSIMD_DEFAULT_ALIGNMENT)
    for (int j = 0; j < NCols; j++) {
      outData[j] = std::exp(inData[j] - maxVal);
      sum += outData[j];
    }
#pragma omp simd aligned(outData : XSIMD_DEFAULT_ALIGNMENT)
    for (int j = 0; j < NCols; j++) {
      outData[j] /= sum;
    }
  }
}
/// [row-softmax]

static void BENCHMARK_NAME(CPP_SIMD_OpenMP)(benchmark::State &state) {

  /// [declare-io]
  aligned_vector<float> Q(SEQUENCE_LENGTH * DM, 1);
  aligned_vector<float> K(SEQUENCE_LENGTH * DM, 1);
  aligned_vector<float> V(SEQUENCE_LENGTH * DM, 1);
  aligned_vector<float> QK(SEQUENCE_LENGTH * SEQUENCE_LENGTH, -1);
  aligned_vector<float> Output(SEQUENCE_LENGTH * DM, -1);
  /// [declare-io]

  for (auto _ : state) {
    /// [scaled-dot-product]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, /*M=*/SEQUENCE_LENGTH,
                /*N=*/SEQUENCE_LENGTH, /*K=*/SEQUENCE_LENGTH,
                /*alpha=*/TEMPERATURE_INV, Q.data(), /*lda=*/SEQUENCE_LENGTH, K.data(),
                /*ldb=*/DM,
                /*beta=*/0, QK.data(), /*ldc=*/DM);

    row_softmax<SEQUENCE_LENGTH, SEQUENCE_LENGTH>(QK.data(), QK.data());

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                /*M=*/SEQUENCE_LENGTH, /*N=*/DM,
                /*K=*/SEQUENCE_LENGTH,
                /*alpha=*/1, QK.data(), /*lda=*/SEQUENCE_LENGTH, V.data(),
                /*ldb=*/DM,
                /*beta=*/0, Output.data(), /*ldc=*/DM);
    /// [scaled-dot-product]

    benchmark::DoNotOptimize(QK.data());
    benchmark::DoNotOptimize(Output.data());
    benchmark::ClobberMemory();
  }
  state.counters["Value"] = Output[0];
  state.counters["QK"]    = QK[0];
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_SIMD_OpenMP));
