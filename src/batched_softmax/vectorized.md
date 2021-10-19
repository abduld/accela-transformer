---
benchmark_name: Accera_Vectorized
cpp_code: src/batched_softmax/accera_vectorized.cpp
accera_code: src/batched_softmax/vectorized.py
---
# Vectorized Accera

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}`.
> The full source code listing of the Accera code generator can be found in  [{{accera_code}} :fas fa-code: ]({{accera_code}}) and the benchmark runner is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).
 
 
As in the [naive](naive.md) implementation, we first need to import the required packages:

[vectorized.py](vectorized.py ':include :type=code python :fragment=import-package')

We then define the input size:

[vectorized.py](vectorized.py ':include :type=code python :fragment=declare-input-size')

Target dependent characteristics can be queried from by creating a package target.

[vectorized.py](vectorized.py ':include :type=code python :fragment=declare-target-dependent-properties')

The initialization function can be defined as follows:

[vectorized.py](vectorized.py ':include :type=code python :fragment=init')

The max function can be defined as follows:

[vectorized.py](vectorized.py ':include :type=code python :fragment=max')

The `exp` function can be defined as follows:

[vectorized.py](vectorized.py ':include :type=code python :fragment=exp')

The reduction function can be defined as follows:

[vectorized.py](vectorized.py ':include :type=code python :fragment=accum')

The normalization function can be defined as follows:

[vectorized.py](vectorized.py ':include :type=code python :fragment=div')

We can combine the above functions into a single one by defining a `softmax` function.
We return a plan along with the inputs from the `softmax` function.

[vectorized.py](vectorized.py ':include :type=code python :fragment=softmax')

We then add and exportethe `softmax` plan.
Once the package is exported, the object file will contain the individual function definitions (`vectorized_init`, `vectorized_max`, `vectorized_exp`, `vectorized_accum`, and `vectorized_div`). 
The object file will also contain the softmax implementation (which we called `vectorized`) which invokes those functions in the sequence defined.

[vectorized.py](vectorized.py ':include :type=code python :fragment=export-package')

## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_vectorized.cpp](accera_vectorized.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_vectorized.cpp](accera_vectorized.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_vectorized.cpp](accera_vectorized.cpp ':include :type=code cpp :fragment=use-function')
 