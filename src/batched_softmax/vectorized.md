---
benchmark_name: Accera_Vectorized
cpp_code: src/batched_softmax/accera_vectorized.cpp
accera_code: src/batched_softmax/vectorized.py
---
# Vectorized Accera

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}`.
> The full source code listing of the Accera code generator can be found in  [{{accera_code}} :fas fa-code: ]({{accera_code}}) and the benchmark runner is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).
 
 

[vectorized.py](vectorized.py ':include :type=code python :fragment=import-package')

[vectorized.py](vectorized.py ':include :type=code python :fragment=declare-input-size')

[vectorized.py](vectorized.py ':include :type=code python :fragment=declare-target-dependent-properties')

[vectorized.py](vectorized.py ':include :type=code python :fragment=init')

[vectorized.py](vectorized.py ':include :type=code python :fragment=max')

[vectorized.py](vectorized.py ':include :type=code python :fragment=exp')

[vectorized.py](vectorized.py ':include :type=code python :fragment=accum')

[vectorized.py](vectorized.py ':include :type=code python :fragment=div')

[vectorized.py](vectorized.py ':include :type=code python :fragment=softmax')

[vectorized.py](vectorized.py ':include :type=code python :fragment=export-package')

## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_vectorized.cpp](accera_vectorized.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_vectorized.cpp](accera_vectorized.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_vectorized.cpp](accera_vectorized.cpp ':include :type=code cpp :fragment=use-function')
