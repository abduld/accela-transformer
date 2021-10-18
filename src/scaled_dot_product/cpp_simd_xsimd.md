---
cpp_code: src/scaled_dot_product/cpp_simd_xsimd.cpp
---
# Vectorized C++ Using XSIMD
<!-- 
[:fas fa-code:](src/scaled_dot_product/cpp_simd_xsimd.cpp) -->

We implement a vectorized scaled dot product in C++ by leveraging the [XSIMD](https://github.com/xtensor-stack/xsimd) library.

[](cpp_simd_xsimd.cpp ':include :type=code cpp :fragment=scaled-dot-product')

With the following inputs and output:

[](cpp_simd_xsimd.cpp ':include :type=code cpp :fragment=declare-io')
 

Where the `row_softmax` function is defined as:

[](cpp_simd_xsimd.cpp ':include :type=code cpp :fragment=row-softmax')