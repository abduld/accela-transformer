import google_benchmark as benchmark

import numpy as np
import torch

N = 2 ** 20

@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def softmax_numpy(state):
    a = np.random.rand(N)
    while state:
      m = a.max()
      res = np.exp(a - m)
      denom = res.sum()
      res = res / denom


@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def softmax_pytorch(state):
    a = torch.randn(N)
    while state:
        res = torch.nn.Softmax(a)


if __name__ == "__main__":
    benchmark.main()