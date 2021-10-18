---
benchmark_names:
    - CPP_SIMD_OpenMP_BatchFirst
    - CPP_SIMD_OpenMP_LengthFirst
    - CPP_SIMD_OpenMP_Mixed
cpp_code: src/batched_softmax/cpp_simd_openmp.cpp
---
# Vectorized C++ Using OpenMP

> [!Note]
> The following shows the implementation of the `{{benchmark_names}}` baseline.
> The full source code listing is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).


For the naive implementation, we consider three different traversal patterns for computing the batched softmax.
The implementation first iterates over the each element in the batch (`BATCH_SIZE` in the outer most loop):


[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=batch-first')


The second iterates over the elements in the loop ( `N` in the outer most loop).
This requires allocation of temporary buffers:


[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=length-first')


The final implementation iterates over the batch size when computing the maximum and performing the normalization, and over the length when computing the exponential.
This implementation also requires allocation of temporary buffers:


[](cpp_simd_openmp.cpp ':include :type=code cpp :fragment=mixed')
