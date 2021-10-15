#include "config.hpp"
#include "utils.hpp"

#include <cblas.h>

template <int NRows, int NCols>
static void row_softmax(float *outData0, const float *inData0) {
  for (int ii = 0; ii < NRows; ii++) {
    const auto inData = inData0 + ii * NCols;
    auto outData      = outData0 + ii * NCols;
    float maxVal      = xsimd::reduce(inData, inData + NCols, inData[0],
                                 [=](const auto &x, const auto &y) { return xsimd::max(x, y); });
    xsimd::transform(inData, inData + NCols, outData,
                     [=](const auto &x) { return xsimd::exp(x - maxVal); });
    float totalVal = xsimd::reduce(outData, outData + NCols, 0.0f);
    xsimd::transform(outData, outData + NCols, outData,
                     [=](const auto &x) { return x / totalVal; });
  }
}

static void BENCHMARK_NAME(CPP_Naive)(benchmark::State &state) {
  using vec = aligned_vector<float>;

  vec Q(SEQUENCE_LENGTH * DM, 1), K(SEQUENCE_LENGTH * DM, 1), V(SEQUENCE_LENGTH * DM, 1);
  vec QK(SEQUENCE_LENGTH * SEQUENCE_LENGTH, 0), Output(SEQUENCE_LENGTH * DM, 0);

  for (auto _ : state) {

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, /*M=*/SEQUENCE_LENGTH, /*N=*/SEQUENCE_LENGTH, /*K=*/SEQUENCE_LENGTH,
                /*alpha=*/TEMPERATURE_INV, Q.data(), /*lda=*/SEQUENCE_LENGTH, K.data(), /*ldb=*/DM,
                /*beta=*/0, QK.data(), /*ldc=*/DM);

    row_softmax<SEQUENCE_LENGTH, SEQUENCE_LENGTH>(QK.data(), QK.data());

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, /*M=*/SEQUENCE_LENGTH, /*N=*/DM,
                /*K=*/SEQUENCE_LENGTH,
                /*alpha=*/TEMPERATURE_INV, QK.data(), /*lda=*/SEQUENCE_LENGTH, V.data(), /*ldb=*/DM,
                /*beta=*/0, Output.data(), /*ldc=*/DM);

    benchmark::DoNotOptimize(QK.data());
    benchmark::DoNotOptimize(Output.data());
    benchmark::ClobberMemory();
  }
}

ADD_BENCHMARK(BENCHMARK_NAME(CPP_Naive));

