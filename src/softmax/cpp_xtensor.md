# Vectorized C++ Using XTensor

We implement a vectorized reductions in C++ by leveraging the [XTensor](https://github.com/xtensor-stack/xtensor) library.


First, compute the max value of the input data.

[](cpp_simd_xtensor.cpp ':include :type=code cpp :fragment=max-val')
 
Compute the sum of exponents.

[](cpp_simd_xtensor.cpp ':include :type=code cpp :fragment=sum-exp')
 
 
Normalize the output.

[](cpp_simd_xtensor.cpp ':include :type=code cpp :fragment=divide')