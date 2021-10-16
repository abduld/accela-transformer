# Naive Accera Implementation

The general idea of a naive

```algorithm
\begin{algorithm} 
\begin{algorithmic} 
\PROCEDURE{NaiveReduction}{$Input$}
    \STATE sum = 0 
    \FOR{$i$ = 0 \TO N} 
        \STATE sum = sum + Input[$i$]
    \ENDFOR

\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```


[naive.drawio](naive.drawio ':include :type=code drawio')

To start, we need to import the Accera package:

[naive.py](naive.py ':include :type=code python :fragment=import-package')

We then define the input size (Which is also the iteration domain) of the problem.
We choose a sufficiently large size to be able to more accurately compare performance across implementation and hide system noise.

[naive.py](naive.py ':include :type=code python :fragment=declare-input-length')

The inputs and output can then be defined based on the size $N$. 
Here we define an `Input` array which is of type `float32` and is of shape `(N,)` --- i.e. a vector of size $N$.
We also define an output array `Sum` which is also of type `float32` but contains a single element --- i.e. a 1-dimensional vector of size 1.

[naive.py](naive.py ':include :type=code python :fragment=declare-inputs')

The nest is then defined. 
Since the iteration domain is between $0$ and $N-1$, we define the nest shape to be a 1-dimensional vector of size $N$.

[naive.py](naive.py ':include :type=code python :fragment=declare-nest')

Use use the `get_indices` function to get the iteration variable used in the nest:

[naive.py](naive.py ':include :type=code python :fragment=get-nest-indices')

As described above, the reduction sums all inputs into a single variable; i.e., for any iteration $i$ we perform `Sum[0] += Input[i]`.
With that, we define the iteration logic function and add it to the nest: 

[naive.py](naive.py ':include :type=code python :fragment=declare-iteration-logic')

We can then create a package and add the `accum_nest` to our package.
Our nest takes $2$ arguments `Sum` and `Input` and we set it's exported C-name to be called `"naive"`.

[naive.py](naive.py ':include :type=code python :fragment=create-package')

We then export the package as a `HAT` package called `"naive"`. 

[naive.py](naive.py ':include :type=code python :fragment=build-package')

The output of the above are two files:
    
- A `naive.hat` file which is a header file containing the declaration of the function.
- A `naive.o` file swhich contains the definition of the function.

Users can use the exported file within their C code by first importing the `naive.hat` file

```cpp
#include "naive.hat"
```

And then using the exported function (passing the expected arguments):

```cpp
static size_t N = 1 << 20;              // declare the problem size
std::vector<float> Input(N, 1.0/N);     // a vector where each element is 1/N
                                        // (the sum is therefore equal to N)
float sum = 0;                          // declare the output
naive(&sum, Input.data());              // invoke the implementation
std::cout << "Sum = " << sum << "\n";   // print the sum
```

