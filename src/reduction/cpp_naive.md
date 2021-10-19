---
benchmark_names:
    - CPP_Naive
    - CPP_Naive_Algorithm
cpp_code: src/reduction/cpp_naive.cpp
---
# Naive C++

> [!Note]
> The following shows the implementation of the `{{benchmark_names}}` baselines.
> The full source code listing is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).


We implement two versions of reductions in pure C++.
The first (`CPP_Naive`) uses a simple loop to perform the accumulation:


[](cpp_naive.cpp ':include :type=code cpp :fragment=simple-loop')

The second (`CPP_Naive_Algorithm`) uses the C++ library function `std::accumulate` to perform the reduction:


[](cpp_naive.cpp ':include :type=code cpp :fragment=std-accumulate')
