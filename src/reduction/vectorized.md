# Vectorized Accera

The following shows how to implement tree reduction using Accera.
There are only a few tweaks that we need to make to the [naive](src/reduction/naive.md) Accera implementation to enable vectorization. 

## Overview

The idea of tree reduction is rather simple.
Suppose we are given a vector of length $N$:

```ditaa
<------ N ------>
+--+--+--+   +--+
|  |  |  |...|  |
+--+--+--+   +--+
``` 

We first partition the input into chunks of size `vector_size` (we use `vector_size=2` in the visualization)



```ditaa
<---------- N/2 ---------->
+--+--+ +--+--+     +--+--+
|  |  | |  |  | ... |  |  |
+--+--+ +--+--+     +--+--+
<- 2 -> <- 2 -> ... <- 2 ->
```

There are $N/2$ of these $2$-element chunks.
This is just a different view of the input vector and does not change the layout of the data.
We then proceed by adding the $2$-element chunks with each other:

```ditaa
+--+--+   +--+--+             +--+--+
|  |  | + |  |  | ...  +  ... |  |  |
+--+--+   +--+--+             +--+--+
```

The result is a $2$-element vector. 

```ditaa
+--+--+ 
|  |  |  
+--+--+ 
```

We finally perform a horizontal add on the $2$-element vector to get the output scalar value:

```ditaa
+--+   +--+ 
|  | + |  |  
+--+   +--+ 
```

The pseudocode of the above strategy can be expressed as:


```algorithm
\begin{algorithm} 
\begin{algorithmic} 
\PROCEDURE{VectorizedReduction}{$Input$}
    \STATE sumVec = \{0, $\ldots$, 0\}
    \FOR{$i$ = 0 \TO $\frac{N}{VecSize}$}  
        \STATE sumVec = sumVec + Input[$i$ * VecSize : $(i+1)$ * VecSize]
    \ENDFOR
    \STATE sum = 0 
    \FOR{$k$ = 0 \TO VecSize} 
        \STATE sum = sum + sumVec[$k$]
    \ENDFOR
    \RETURN sum
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```

## Implementation

Writing vectorized reduction in Accera follows from the pseudocode above.
First, we need to import the `accera` package:

[](vectorized.py ':include :type=code python :fragment=import-package')

We then define some constants which are derived from the number of vector bytes on the host system.
The system which the benchmark is run on has AVX-2 support, therefore the vector bytes is `32`.
Since the byte count of a single-precision float is `4`, we divide the vector bytes by `4` to get the number of single-precision elements in an AVX-2 vector (which is `8` single-precision floating point values).
using the `vector_size`, we derive two other constants (`vector_units` and `split_size`) which we picked based on properties of the host  system.
The reader is encouraged to try other multiples of `vector_size` for these parameters and observe performance differences.

[](vectorized.py ':include :type=code python :fragment=declare-target-dependent-properties')

As in the naive case, we define our inputs

[](vectorized.py ':include :type=code python :fragment=declare-inputs')

We also define an auxillary array which has the same number of elements as the `split_size`.
The `SumVec` will be used to store intermediate sums during the tree reduction.

[](vectorized.py ':include :type=code python :fragment=declare-input-vec')

We then define the loop nest.
The loop nest has a $2$-dimensional iteration space.
The outer dimension corresponds to the number of splits we are processing and the inner dimension corresponds to the size of the split.
We then perform the computation.
Since Input is a $1$-dimensional vector, we need to map the $2$-dimensional indices into the vector (i.e., the index computation `i * split_size + j`).

[](vectorized.py ':include :type=code python :fragment=declare-vector-reduction-iteration-logic')

The above is equivalent to the following Python loop:

```python
for i in range(N // split_size):
    for j in range(split_size):
        SumVec[j] += Input[i * split_size + j]
```

We also define the final reduction nest.
This nest adds the elements of `SumVec` together.

[](vectorized.py ':include :type=code python :fragment=declare-horizontal-reduction-iteration-logic')

We create a schedule of the two nests.

[](vectorized.py ':include :type=code python :fragment=create-two-schedules')

We then perform concatenation fusing (i.e., fusing with `partial=0`).

[](vectorized.py ':include :type=code python :fragment=fuse-two-schedules')

The above would effectively concatenate the two loops generating the following equivalent Python code:

```python
for i in range(N // split_size):
    for j in range(split_size):
        SumVec[j] += Input[i * split_size + j]
for k in range(split_size):
    Sum[0] += SumVec[k]
```

Querying for the indices of the fused schedule returns:

 - The fused dimension `f` 
 - The indices of the `vec_accum_schedule` (`i,j`)
 - The index of `finalize_accum_schedule` (`k`)



[](vectorized.py ':include :type=code python :fragment=get-fused-schedule-indices')

We can use the indices to perform a split based on the number of vector units:

[](vectorized.py ':include :type=code python :fragment=split-index-by-vector-units')

I.e., the above would transform our code into:

```python
for i in range(0, N // split_size, vector_units):
    for ii in range(i, i + vector_units):
        for j in range(split_size):
            SumVec[j] += Input[ii * split_size + j]
for k in range(split_size):
    Sum[0] += SumVec[k]
```

We use the `fused_schedule` to create an action plan

[](vectorized.py ':include :type=code python :fragment=create-fused-action-plan')

To increase performance, we apply vectorization and unrolling on the indices

[](vectorized.py ':include :type=code python :fragment=optimize-indices')

We then add the fused plan to our package as a function called `vectorized`

[](vectorized.py ':include :type=code python :fragment=create-package')

We export our package as an object file called `vectorized`.

[](vectorized.py ':include :type=code python :fragment=export-package')

## Usage

The package can then be used within the C code base. 
Here we benchmark the `vectorized` function:

[](accera_vectorized.cpp ':include :type=code cpp')