---
benchmark_name: CPP_SIMD_OpenMP
cpp_code: src/scaled_dot_product/cpp_simd_openmp.cpp
---

# Vectorized C++ Using OpenMP

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}` baseline.
> The full source code listing is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}). 

We implement a vectorized scaled dot product in C++ using OpenMP annotations.

[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=scaled-dot-product')
 

With the following inputs and output:

[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=declare-io')
 
Where the `row_softmax` function is defined as:

[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=row-softmax')