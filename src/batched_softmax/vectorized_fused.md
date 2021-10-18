# Vectorized Accera

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=import-package')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=declare-input-size')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=declare-target-dependent-properties')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=init')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=max')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=exp')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=accum')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=div')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=softmax')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=export-package')

## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_vectorized_fused.cpp](accera_vectorized_fused.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_vectorized_fused.cpp](accera_vectorized_fused.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_vectorized_fused.cpp](accera_vectorized_fused.cpp ':include :type=code cpp :fragment=use-function')
