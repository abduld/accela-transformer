# Vectorized C++ Using XSIMD

We implement a vectorized reductions in C++ by leveraging the [XSIMD](https://github.com/xtensor-stack/xsimd) library.


First, compute the max value of the input data.

[](cpp_simd_xsimd.cpp ':include :type=code cpp :fragment=max-val')
 
Compute the sum of exponents.

[](cpp_simd_xsimd.cpp ':include :type=code cpp :fragment=sum-exp')
 
 
Normalize the output.

[](cpp_simd_xsimd.cpp ':include :type=code cpp :fragment=divide')