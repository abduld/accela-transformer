---
benchmark_name: Accera_Naive
cpp_code: src/batched_softmax/accera_vectorized_fused.cpp
accera_code: src/batched_softmax/vectorized_fused.py
---
# Naive Accera



> [!ATTENTION]
> Implement me.

> [!Note]
> The following shows the implementation of the `{{benchmark_name}}`.
> The full source code listing of the Accera code generator can be found in  [{{accera_code}} :fas fa-code: ]({{accera_code}}) and the benchmark runner is found in [{{cpp_code}} :fas fa-code: ]({{cpp_code}}).
 

The pseudocode of the naive implementation is:

```algorithm
\begin{algorithm} 
\begin{algorithmic} 
\PROCEDURE{BatchedSoftmax}{$Input$}
    \STATE maxVal = -$\infty$
    \STATE denom = $0$
    \FOR{$b$ = 0 \TO \texttt{BATCH\_SIZE}} 
        \FOR{$m$ = 0 \TO \texttt{N}} 
            \STATE maxVal = $max$(maxVal[b], Input[bm, m])
        \ENDFOR 
        \FOR{$i$ = 0 \TO \texttt{N}} 
            \STATE Output[b, i] = $e^{\text{Input[b, i]} - \text{maxVal}}$
        \ENDFOR 
        \FOR{$a$ = 0 \TO \texttt{N}} 
            \STATE demon = denom + Output[b, a]
        \ENDFOR 
        \FOR{$j$ = 0 \TO \texttt{N}} 
            \STATE Output[b,j] = $\frac{\text{Output[b,j]}}{\text{denom}}$ 
        \ENDFOR 
    \ENDFOR  
    \RETURN Output
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```


A vectorized and fused batched Softmax implementation in Accera follows from the above pseudocode and requires minor tweaks to the schedule.
We use the [Accera naive implementation](naive.py) as basis.
The code does not change until we create the fused schedule:

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=naive')


We fuse all the schedules into a single schedule and construct an action plan from that.

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=fuse')


[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=schedule')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=vectorize')

[vectorized_fused.py](vectorized_fused.py ':include :type=code python :fragment=export-package')


## Usage

The package can then be used within our C code base.
To do so, we first need to import the HAT package created:

[accera_vectorized_fused.cpp](accera_vectorized_fused.cpp ':include :type=code cpp :fragment=import-hat')

We then declare our inputs and outputs:

[accera_vectorized_fused.cpp](accera_vectorized_fused.cpp ':include :type=code cpp :fragment=declare-io')

We then can use the exported function within our C code:

[accera_vectorized_fused.cpp](accera_vectorized_fused.cpp ':include :type=code cpp :fragment=use-function')
