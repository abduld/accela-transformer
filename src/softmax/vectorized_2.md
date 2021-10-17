# Vectorized 2 Accera

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

[robocode_vectorized_2.cpp](robocode_vectorized_2.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[robocode_vectorized_2.cpp](robocode_vectorized_2.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[robocode_vectorized_2.cpp](robocode_vectorized_2.cpp ':include :type=code cpp :fragment=use-function')