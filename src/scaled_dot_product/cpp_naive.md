---
benchmark_name: CPP_Naive
cpp_code: src/scaled_dot_product/cpp_naive.cpp
---
# Naive C++

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}` baseline.
> The full source code listing is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).

We implement a naive C++ scaled dot product as

[](cpp_naive.cpp ':include :type=code cpp :fragment=scaled-dot-product')
 
With the following inputs and output:

[](cpp_naive.cpp ':include :type=code cpp :fragment=declare-io')
 

Where the `row_softmax` function is defined as:

[](cpp_naive.cpp ':include :type=code cpp :fragment=row-softmax')