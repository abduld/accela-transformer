# Vectorized C++ Using OpenMP

We implement a vectorized Softmax in C++ by leveraging the OpenMP annotations.

First, compute the max value of the input data.

[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=max-val')
 
Compute the sum of exponents.

[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=sum-exp')
 
Normalize the output.

[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=divide')