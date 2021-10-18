# Scaled Dot Product

A Scaled Dot Product is a primitive operation in a transformer which leverages both the Batched Softmax along with Matmul.
The pseudocode of the naive implementation is:

```algorithm
\begin{algorithm} 
\begin{algorithmic} 
\PROCEDURE{ScaledDotProduct}{Q, K, V}
    \RETURN Softmax($\frac{\text{Q}}{\sqrt{\text{DK}}}$ $\cdot$ $\text{K}^T$) $\cdot$ V
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```

Where $\text{A} \cdot \text{B}$ multiplies the matrix A with B.