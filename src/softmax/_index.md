# Softmax <!-- {docsify-ignore-all} -->

This section shows how to implement [Softmax](https://en.wikipedia.org/wiki/Softmax_function)  using Accera.
Softmax of a given a vector $V = \left [ a_{0}, a_{1}, \cdots, a_{n} \right ]$ is defined by $\displaystyle \frac{e^{a_i}}{\sum_{i =0}^n e^{a_i}}$.
For numerical stability, however, we define Softmax as $\displaystyle \frac{e^{a_i - max(V)}}{\sum_{i =0}^n e^{a_i - max(V)}}$.


