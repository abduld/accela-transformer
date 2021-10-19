# Two Pass Softmax

https://arxiv.org/pdf/2001.04438.pdf


```algorithm
\begin{algorithm}[H]
    \begin{algorithmic}
        \Function {SoftmaxTwoPass}{$X$, $Y$}
            \State $N \gets \Call{Length}{X}$
            \State $m_{sum} \gets 0$
            \State $n_{sum} \gets -\infty$
            \ForAll{$1 \leq i \leq N$}
                \State $m_i, n_i \gets \Call{ExtExp}{X_i}$
                \Comment{Pass 1: read X}
                \State $n_{max} \gets \Call{Max}{n_i, n_{sum}}$
                \State $m_{sum} \gets m_i \cdot 2^{n_i - n_{max}} + m_{sum} \cdot 2^{n_{sum} - n_{max}}$
                \State $n_{sum} \gets n_{max}$
            \EndFor
            \State $\lambda_{sum} \gets \sfrac{1}{m_{sum}}$
            \ForAll{$1 \leq i \leq N$}
                \State $m_i, n_i \gets \Call{ExtExp}{X_i}$
                \Comment{Pass 2: read X, write Y}
                \State $Y_i \gets m_i \cdot \lambda_{sum} \cdot 2^{n_i - n_{sum}}$
            \EndFor
        \EndFunction
    \end{algorithmic}
    \caption{The Two-Pass softmax algorithm. $\textrm{ExtExp}$ denotes an exponential function that produce a pair $(m, n)$ of floating-point values.}
    \label{algo:two-pass}
\end{algorithm}
```