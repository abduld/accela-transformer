---
benchmark_names:
    - CPP_Naive_BatchFirst
    - CPP_Naive_LengthFirst
    - CPP_Naive_Mixed
cpp_code: src/batched_softmax/cpp_naive.cpp
---
# Naive C++

> [!Note]
> The following shows the implementation of the `{{benchmark_names}}` baseline.
> The full source code listing is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).


For the naive implementation, we consider three different traversal patterns for computing the batched softmax.
The implementation first iterates over the each element in the batch (`BATCH_SIZE` in the outer most loop):


[](cpp_naive.cpp ':include :type=code cpp :fragment=batch-first')


The second iterates over the elements in the loop ( `N` in the outer most loop).
This requires allocation of temporary buffers:


[](cpp_naive.cpp ':include :type=code cpp :fragment=length-first')


The final implementation iterates over the batch size when computing the maximum and performing the normalization, and over the length when computing the exponential.
This implementation also requires allocation of temporary buffers:


[](cpp_naive.cpp ':include :type=code cpp :fragment=mixed')
