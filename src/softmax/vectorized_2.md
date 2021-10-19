---
benchmark_name: Accera_Vectorized2
cpp_code: src/softmax/accera_vectorized_2.cpp
accera_code: src/softmax/vectorized_2.py
---
# Vectorized Accera 2

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}`.
> The full source code listing of the Accera code generator can be found in  [{{accera_code}} :fas fa-code: ]({{accera_code}}) and the benchmark runner is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).
 
 
As in the [naive](naive.md) implementation, we first need to import the required packages:

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=import-package')

We then define the input size:

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=declare-input-length')

Target dependent characteristics can be queried from by creating a package target.

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=declare-target')

We define our package which will be used throughout our program.

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=declare-package')

As done previously, the input and output arrays are defined along with the auxillary temporaries.

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=declare-arrays')

The `max` operation schedule can be defined and vectorized.

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=max')

We define the `exp` nest.

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=exp')


The `accum` operation schedule is defined, but we do not perform vectorization on it.

> [!ATTENTION]
> The `accum` nest can be vectorized as shown in the [vectorized reduction](../reduction/vectorized.md) case study.

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=accum')

We finally define the normalization nest.

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=div')

We invoke the above functions to add the to the package.

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=add-to-package')

Finally, we export the package.

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=export-package')

## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_vectorized_2.cpp](accera_vectorized_2.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_vectorized_2.cpp](accera_vectorized_2.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_vectorized_2.cpp](accera_vectorized_2.cpp ':include :type=code cpp :fragment=use-function')