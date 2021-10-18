# Batched Softmax

A Batched Softmax is an extension to Softmax.
Batched Softmax applies the Softmax function from each batch in an input Tensor.
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
