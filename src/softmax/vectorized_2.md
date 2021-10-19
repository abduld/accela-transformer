---
benchmark_name: Accera_Vectorized2
cpp_code: src/softmax/accera_vectorized_2.cpp
accera_code: src/softmax/vectorized_2.py
---
# Vectorized Accera 2

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}`.
> The full source code listing of the Accera code generator can be found in  [{{accera_code}} :fas fa-code: ]({{accera_code}}) and the benchmark runner is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).
 
 

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=import-package')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=declare-input-length')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=declare-target')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=declare-package')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=declare-arrays')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=max')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=exp')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=exp')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=div')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=add-to-package')

[vectorized_2.py](vectorized_2.py ':include :type=code python :fragment=export-package')

## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_vectorized_2.cpp](accera_vectorized_2.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_vectorized_2.cpp](accera_vectorized_2.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_vectorized_2.cpp](accera_vectorized_2.cpp ':include :type=code cpp :fragment=use-function')