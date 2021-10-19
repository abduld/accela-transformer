---
benchmark_name: CPP_SIMD_OpenMP
cpp_code: src/reduction/cpp_simd_openmp.cpp
---
# Vectorized C++ Using OpenMP

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}` baseline.
> The full source code listing is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}). 

We implement a vectorized reductions in C++ using OpenMP annotations.



[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=reduce')
 