# Naive Accera

To implement a basic Softmax in Accera, we need to import the required packages.
We import that `math` package to use the `inf` ($\infty$) constant.

[](naive.py ':include :type=code python :fragment=import-package')


We then define the input vector size (Which is also the iteration domain) of the problem.
 
[](naive.py ':include :type=code python :fragment=declare-input-length')
 
We declare the input and output arrays.
Both are a $1$-dimensional vector of length $N$.

[](naive.py ':include :type=code python :fragment=declare-input-output-arrays')

We declare auxiliary temporary vectors that are used in the computation.
Both the `Denom` and `MaxVal` are scalars (i.e., defined as $1$-dimensional vectors of length $1$).

[](naive.py ':include :type=code python :fragment=declare-temp-arrays')

We create a single-iteration nest which initializes the two temporary scalars with the initial values.
Note that the initialization of `Denom` here is not necessary, since the semantics of Accera dictate that temporary arrays are initialized to $0$.
For completeness, we initialize `Denom`.

[](naive.py ':include :type=code python :fragment=init')

A nest to compute the maximum element in the `Input` vector is then created.

[](naive.py ':include :type=code python :fragment=max')

We compute the exponential of each input element when subtracted from the maximum value; i.e., we are computing the equation $\displaystyle \text{Output}[i] e^{\text{Input}[i] - max(V)}$.
We store the result in the output vector.

[](naive.py ':include :type=code python :fragment=exp')

We compute the sum of the output vector; i.e., we are computing the equation $\displaystyle \text{Output}[i] = \sum_{i =0}^n e^{\text{Input}[i] - max(V)}$.

[](naive.py ':include :type=code python :fragment=accum')

We then divide the output vector by the sum; i.e., we are computing the equation $\displaystyle \text{Output}[i] = \frac{e^{\text{Input}[i] - max(V)}}{\sum_{i =0}^n e^{\text{Input}[i] - max(V)}}$.

[](naive.py ':include :type=code python :fragment=divide')

We fuse all the schedules created:

[](naive.py ':include :type=code python :fragment=fuse')

We then export the schedule as function called `naive` into an object file called `naive`.

[](naive.py ':include :type=code python :fragment=export')


## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=use-function')