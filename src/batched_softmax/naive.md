# Naive Accera

[naive.py](naive.py ':include :type=code python :fragment=import-package')

[naive.py](naive.py ':include :type=code python :fragment=declare-input')

[naive.py](naive.py ':include :type=code python :fragment=init')

[naive.py](naive.py ':include :type=code python :fragment=max')

[naive.py](naive.py ':include :type=code python :fragment=exp')

[naive.py](naive.py ':include :type=code python :fragment=accum')

[naive.py](naive.py ':include :type=code python :fragment=div')

[naive.py](naive.py ':include :type=code python :fragment=fuse')

[naive.py](naive.py ':include :type=code python :fragment=export-package')


## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=use-function')
