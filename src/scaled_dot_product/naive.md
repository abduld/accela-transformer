---
benchmark_names:
    - Accera_Naive_SEQ
    - Accera_Naive
cpp_code: src/scaled_dot_product/accera_naive.cpp
accera_code: src/scaled_dot_product/naive.py
---

# Naive Accera

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}` baselines.
> The full source code listing is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).

We first import our package.
We leverage the same [vectorized Softmax implementation discussed previously](../batched_softmax/vectorized.md).

[naive.py](naive.py ':include :type=code python :fragment=import-package')

We declare our input sizes.

[naive.py](naive.py ':include :type=code python :fragment=declare-input-size')

As described in the [overview](_index.md), the scaled dot product takes the arrays $Q$, $K$, and $V$ as input.

[naive.py](naive.py ':include :type=code python :fragment=declare-input-arrays')

We also declare the temporary arrays:

[naive.py](naive.py ':include :type=code python :fragment=declare-tmp-arrays')

We create our target and package:

[naive.py](naive.py ':include :type=code python :fragment=declare-package')

We then construct our `gemm` and `softmax` plans for the different input sizes and add them to the package.

[naive.py](naive.py ':include :type=code python :fragment=add-functions-to-package')

We define the `scaled_dot_product_attention` function which invokes the functions defined above.
We add that to the package as well.

[naive.py](naive.py ':include :type=code python :fragment=define-dispatch-function')

Finally, we export the package.

[naive.py](naive.py ':include :type=code python :fragment=export-package')


## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code as either a sequence of function calls:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=call-function-sequence')


We can also call the  `scaled_dot_product_attention` function directly:



[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=call-scaled-dot-product')