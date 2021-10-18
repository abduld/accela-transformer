---
benchmark_name: Accera_Naive
cpp_code: src/batched_softmax/accera_naive.cpp
accera_code: src/batched_softmax/naive.py
---
# Naive Accera

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}`.
> The full source code listing of the Accera code generator can be found in  [{{accera_code}} :fas fa-code: ]({{accera_code}}) and the benchmark runner is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).
 

The pseudocode of the naive implementation is:

```algorithm
\begin{algorithm} 
\begin{algorithmic} 
\PROCEDURE{BatchedSoftmax}{$Input$}
    \STATE maxVal[\texttt{BATCH\_SIZE}] = \{-$\infty$, $\ldots$, -$\infty$\}
    \STATE denom[\texttt{BATCH\_SIZE}] = \{$0$, $\ldots$, $0$\}
    \FOR{$bm$ = 0 \TO \texttt{BATCH\_SIZE}} 
        \FOR{$m$ = 0 \TO \texttt{N}} 
            \STATE maxVal[bm] = $max$(maxVal[bm], Input[bm, m])
        \ENDFOR 
    \ENDFOR 
    \FOR{$bi$ = 0 \TO \texttt{BATCH\_SIZE}} 
        \FOR{$i$ = 0 \TO \texttt{N}} 
            \STATE Output[bi, i] = $e^{\text{Input[bi, i]} - \text{maxVal[bi]}}$
        \ENDFOR 
    \ENDFOR 
    \FOR{$ba$ = 0 \TO \texttt{BATCH\_SIZE}} 
        \FOR{$a$ = 0 \TO \texttt{N}} 
            \STATE demon[ba] = denom[ba] + Output[ba, a]
        \ENDFOR 
    \ENDFOR 
    \FOR{$bj$ = 0 \TO \texttt{BATCH\_SIZE}} 
        \FOR{$j$ = 0 \TO \texttt{N}} 
            \STATE Output[bj,j] = $\frac{\text{Output[bj,j]}}{\text{denom[bj]}}$ 
        \ENDFOR 
    \ENDFOR  
    \RETURN Output
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```

A batched Softmax implementation in Accera follows from the above pseudocode.
First, we need to import the required packages.

[naive.py](naive.py ':include :type=code python :fragment=import-package')

We then define the problem size, input and output arrays, and the auxillary temporary arrays.

[naive.py](naive.py ':include :type=code python :fragment=declare-input')

We initialize the values in the temporary arrays.

[naive.py](naive.py ':include :type=code python :fragment=init')

We compute the maximum value of the inputs within each batch.

[naive.py](naive.py ':include :type=code python :fragment=max')

We compute the exponential value and store the results in the `Output` array.

[naive.py](naive.py ':include :type=code python :fragment=exp')

We accumulate the values of `Output` within each batch.

[naive.py](naive.py ':include :type=code python :fragment=accum')

We normalize the `Output` using the accumulated values.

[naive.py](naive.py ':include :type=code python :fragment=div')

We fuse all the schedules into a single schedule and construct an action plan from that.

[naive.py](naive.py ':include :type=code python :fragment=fuse')

We then export the plan as a `naive` named function in `naive.o` object file.  

[naive.py](naive.py ':include :type=code python :fragment=export-package')


## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_naive.cpp](accera_naive.cpp ':include :type=code cpp :fragment=use-function')
