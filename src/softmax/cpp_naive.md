# Naive C++

We implement a Softmax in plain C++.

First, compute the max value of the input data.

[](cpp_naive.cpp ':include :type=code cpp :fragment=max-val')
 
Compute the sum of exponents.

[](cpp_naive.cpp ':include :type=code cpp :fragment=sum-exp')
 
Normalize the output.

[](cpp_naive.cpp ':include :type=code cpp :fragment=divide')