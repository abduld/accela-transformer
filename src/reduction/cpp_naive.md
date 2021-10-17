# Naive C++

We implement two versions of reductions in pure C++.
The first (`CPP_Naive`) uses a simple loop to perform the accumulation:


[](cpp_naive.cpp ':include :type=code cpp :fragment=simple-loop')

The second (`CPP_Naive_Algorithm`) uses the C++ library function `std::accumulate` to perform the reduction:


[](cpp_naive.cpp ':include :type=code cpp :fragment=std-accumulate')
