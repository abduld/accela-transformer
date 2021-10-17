# Softmax <!-- {docsify-ignore-all} -->

This section shows how to implement [Softmax](https://en.wikipedia.org/wiki/Softmax_function)  using Accera.
Softmax of a given a vector $V = \left [ a_{0}, a_{1}, \cdots, a_{n} \right ]$ is defined by $\displaystyle \frac{e^{a_i}}{\sum_{i =0}^n e^{a_i}}$.
For numerical stability, however, we define Softmax as $\displaystyle \frac{e^{a_i - max(V)}}{\sum_{i =0}^n e^{a_i - max(V)}}$.
The implementation pseudocode is:

```algorithm
\begin{algorithm} 
\begin{algorithmic} 
\PROCEDURE{Softmax}{$Input$}
    \STATE maxVal = -$\infty$
    \STATE denom = 0 
    \FOR{$i$ = 0 \TO N} 
        \STATE maxVal = max(max, Input[$i$])
    \ENDFOR 
    \FOR{$i$ = 0 \TO N} 
        \STATE Output[$i$] = $e^{\text{Input}[i] - maxVal}$
        \STATE denom = denom + Output[$i$]
    \ENDFOR
    \FOR{$i$ = 0 \TO N} 
        \STATE Output[$i$] = $\frac{\text{Output}[i]}{\text{denom}}$
    \ENDFOR
    \RETURN Output
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
