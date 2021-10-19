---
benchmark_name: Accera_Vectorized
cpp_code: src/softmax/accera_vectorized.cpp
accera_code: src/softmax/vectorized.py
---
# Vectorized Accera

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}`.
> The full source code listing of the Accera code generator can be found in  [{{accera_code}} :fas fa-code: ]({{accera_code}}) and the benchmark runner is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).
 
 

We build upon our [naive Softmax](naive.md) implementation to develop a vectorized version of the same algorithm.
Again, we import the required libraries:

[vectorized.py](vectorized.py ':include :type=code python :fragment=import-package')

We also use an internal exponential function that is more amendable to be vectorized.

[vectorized.py](vectorized.py ':include :type=code python :fragment=use-fast-exp')

We then declare the input size.
As before, we choose a sufficiently large size to observe performance differences.

[vectorized.py](vectorized.py ':include :type=code python :fragment=declare-input-length')

We declare our target and `vector_size`. 
The `vector_bytes` specifies the width of the vector unit on the target system.
So, on an AVX-2 system, `vector_bytes` is 32-bytes (corresponding to 256-bit registers) and therefore can hold at most 8 single-precision floating point numbers.

[vectorized.py](vectorized.py ':include :type=code python :fragment=declare-target-depdendent-params')

As in the [naive](naive.md) case, we define our input and output arrays:

[vectorized.py](vectorized.py ':include :type=code python :fragment=declare-input-output-arrays')

We also define our auxillary temporary arrays.

[vectorized.py](vectorized.py ':include :type=code python :fragment=declare-temp-arrays')

Not much changes in defining the initialization schedule.
This nest just initializes the `MaxVal` and `Denom` scalars.

[vectorized.py](vectorized.py ':include :type=code python :fragment=init')

The `max_nest` function computes the max value of all elements in the `Input` array.

[vectorized.py](vectorized.py ':include :type=code python :fragment=max')

We use the `fast_exp_mlas` function to compute the exponential values of the input.

[vectorized.py](vectorized.py ':include :type=code python :fragment=exp')


> [!TIP]
> The user is encouraged to try using `acc.exp` in place of the `fast_exp_mlas` function and observe the performance difference.

We sum he exponentials.

[vectorized.py](vectorized.py ':include :type=code python :fragment=accum')

We then normalize the output using the sum.

[vectorized.py](vectorized.py ':include :type=code python :fragment=divide')

Finally, we fuse all the schedules.

[vectorized.py](vectorized.py ':include :type=code python :fragment=fuse')

Until now there has been no change to our implementation over the [naive](naive.md) implementation.
We now apply optimizations on the schedules creates.
First, we need to get a handle to the indices created by the fused schedule:

[vectorized.py](vectorized.py ':include :type=code python :fragment=fuse-indices')

We split the indices to prepare for vectorization. 

[vectorized.py](vectorized.py ':include :type=code python :fragment=fused-schedule-split')

We then reorder the schedule to facilitate unrolling.
Since the indices `i` and `a` (corresponding to computing the exponential and aggregating the results) can be performed in parallel, we reorder them so that the inner most loops follow each other.

[vectorized.py](vectorized.py ':include :type=code python :fragment=fused-schedule-reorder')

We create the plan from the scheulde.

[vectorized.py](vectorized.py ':include :type=code python :fragment=fused-plan')

We perform the unrolling and vectorization.


[vectorized.py](vectorized.py ':include :type=code python :fragment=fused-plan-unroll-vectorize')

> [!TIP]
> The user is encouraged to try using different unrolling and vectorization configurations.
> Some operations might not benefit from vectorization.
> For example, unrolling the `aa` index.


Finally, we export the package.

[vectorized.py](vectorized.py ':include :type=code python :fragment=export-package')

## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_vectorized.cpp](accera_vectorized.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_vectorized.cpp](accera_vectorized.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_vectorized.cpp](accera_vectorized.cpp ':include :type=code cpp :fragment=use-function')


