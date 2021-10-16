# Benchmark 


```sh
build/bin/reduction  --benchmark_format=csv --benchmark_out=../_assets/results/reduction_amd_epic.csv --benchmark_out_format=csv
```


---------------------------------------------------------------
Benchmark                               Time             CPU   
---------------------------------------------------------------
reduction_Robocode_Vectorized        57.9 us         57.9 us   
reduction_Robocode_Naive              964 us          964 us   
reduction_CPP_XSIMD                   117 us          117 us   
reduction_CPP_SIMD_OpenMP             117 us          117 us   
reduction_CPP_Naive                  1241 us         1241 us   
reduction_CPP_Naive_Algorithm        1238 us         1238 us   
reduction_CPP_Naive_Memcpy            140 us          140 us   
reduction_CPP_Naive_Memcpy_2          144 us          144 us   
reduction_CPP_Memcpy                 95.8 us         95.8 us   
reduction_CPP_Memcpy_XSIMD_2          140 us          140 us   
reduction_CPP_Memcpy_Intrinsic       98.9 us         98.9 us   