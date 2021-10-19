# Benchmark 


```bash
./bin/reduction  --benchmark_out=../_assets/benchmark_results/reduction_amd_epic.csv --benchmark_out_format=csv
```

 
Benchmark Name                        | Time (us)
--------------------------------------|-----------
[Accera_Naive](naive.md)            | 970.427
[Accera_Vectorized](vectorized.md)  | **79.934**
[CPP_Naive](cpp_naive.md)             | 1242.72
[CPP_Naive_Algorithm](cpp_naive.md)   | 1242.29
[CPP_SIMD_OpenMP](cpp_simd_openmp.md) | 116.873
[CPP_XSIMD](cpp_simd_xsimd.md)        | 117.362
