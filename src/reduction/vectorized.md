# Vectorized Accera Implementation

This shows how to implement tree reduction using Accera.
There are only a few tweaks that we need to make to the [naive](naive.md) Accera implementation to enable vectorization. 


The idea of tree reduction is rather simple.
Suppose we are given a vector of length $N$:

```ditaa
<------ N ------>
+--+--+--+=--+--+
|  |  |  |...|  |
+--+--+--+---+--+
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
+--+--+   +--+--+           +--+--+
|  |  | + |  |  | ... + ... |  |  |
+--+--+   +--+--+           +--+--+
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


Writing vectorized reduction in Accera follows from the outline above.
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


[](vectorized.py ':include :type=code python :fragment=declare-vector-reduction-iteration-logic')

[](vectorized.py ':include :type=code python :fragment=declare-horizontal-reduction-iteration-logic')

[](vectorized.py ':include :type=code python :fragment=create-two-schedules')

[](vectorized.py ':include :type=code python :fragment=fuse-two-schedules')


```ditaa
+--------+   +-------+    +-------+
|        +---+ ditaa +--> |       |
|  Text  |   +-------+    |diagram|
|Document|   |!magic!|    |       |
|     {d}|   |       |    |       |
+---+----+   +-------+    +-------+
    :                         ^
    |       Lots of work      |
    +-------------------------+
```

```ditaa
    0               n
   +--+=----+--+--+--+
   |  | ... |  |  |\0|
   +--+-----+--+--+--+
     ^        ^
     |        |
     |        +-- q moves from the
     |            end to the start
     p moves from
       start to the end
```
 

[](vectorized.py ':include :type=code python :fragment=get-fused-schedule-indices')

[](vectorized.py ':include :type=code python :fragment=split-index-by-vector-units')

[](vectorized.py ':include :type=code python :fragment=create-fused-action-plan')

[](vectorized.py ':include :type=code python :fragment=optimize-indices')

[](vectorized.py ':include :type=code python :fragment=create-package')

[](vectorized.py ':include :type=code python :fragment=export-package')