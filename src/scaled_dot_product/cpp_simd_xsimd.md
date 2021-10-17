# Vectorized C++ Using XSIMD

We implement a vectorized scaled dot product in C++ by leveraging the [XSIMD](https://github.com/xtensor-stack/xsimd) library.

[](cpp_simd_xsimd.cpp ':include :type=code cpp :fragment=scaled-dot-product')
 

Where the `row_softmax` function is defined as:

[](cpp_simd_xsimd.cpp ':include :type=code cpp :fragment=row-softmax')