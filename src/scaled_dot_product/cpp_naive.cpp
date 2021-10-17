#include "config.hpp"

#include <cblas.h>

/// [row-softmax]
template <int NRows, int NCols>
static void row_softmax(float *outData0, const float *inData0) {
  for (int i = 0; i < NRows; i++) {
    const auto inData = inData0 + i * NCols;
    auto outData      = outData0 + i * NCols;
    const float max   = *std::max_element(inData, inData + NCols);
    float denominator(0);
    for (int j = 0; j < NCols; j++) {
      outData[j] = std::exp(inData[j] - max);
      denominator += outData[j];
    }
    for (int j = 0; j < N; j++) {
      outData[j] /= denominator;
    }
  }
}
/// [row-softmax]

static void BENCHMARK_NAME(CPP_Naive)(benchmark::State &state) {

  aligned_vector<float> Q(SEQUENCE_LENGTH * DM, 1), K(SEQUENCE_LENGTH * DM, 1),
      V(SEQUENCE_LENGTH * DM, 1);
  aligned_vector<float> QK(SEQUENCE_LENGTH * SEQUENCE_LENGTH, -1), Output(SEQUENCE_LENGTH * DM, -1);

  for (auto _ : state) {
    /// [scaled-dot-product]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, /*M=*/SEQUENCE_LENGTH,
                /*N=*/SEQUENCE_LENGTH, /*K=*/SEQUENCE_LENGTH,
                /*alpha=*/TEMPERATURE_INV, Q.data(), /*lda=*/SEQUENCE_LENGTH, K.data(), /*ldb=*/DM,
                /*beta=*/0, QK.data(), /*ldc=*/DM);

    row_softmax<SEQUENCE_LENGTH, SEQUENCE_LENGTH>(QK.data(), QK.data());

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, /*M=*/SEQUENCE_LENGTH, /*N=*/DM,
                /*K=*/SEQUENCE_LENGTH,
                /*alpha=*/1, QK.data(), /*lda=*/SEQUENCE_LENGTH, V.data(), /*ldb=*/DM,
                /*beta=*/0, Output.data(), /*ldc=*/DM);
    /// [scaled-dot-product]

    benchmark::DoNotOptimize(QK.data());
    benchmark::DoNotOptimize(Output.data());
    benchmark::ClobberMemory();
  }
  state.counters["Value"] = Output[0];
  state.counters["QK"]    = QK[0];
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive));
