#include "config.hpp"

#include <cblas.h>

/// [row-softmax]
template <int NRows, int NCols>
static void row_softmax(float *outData0, const float *inData0) {
  for (int i = 0; i < NRows; i++) {
    const auto inData = inData0 + i * NCols;
    auto outData      = outData0 + i * NCols;
    float maxVal =
        xsimd::reduce(inData, inData + NCols, inData[0],
                      [=](const auto &x, const auto &y) { return xsimd::max(x, y); });
    xsimd::transform(inData, inData + NCols, outData,
                     [=](const auto &x) { return xsimd::exp(x - maxVal); });
    float totalVal = xsimd::reduce(outData, outData + NCols, 0.0f);
    xsimd::transform(outData, outData + NCols, outData,
                     [=](const auto &x) { return x / totalVal; });
  }
}
/// [row-softmax]

static void BENCHMARK_NAME(CPP_XSIMD)(benchmark::State &state) {
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  /// [set-num-threads]
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
  /// [set-num-threads]
  

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

ADD_BENCHMARK(BENCHMARK_NAME(CPP_XSIMD));
