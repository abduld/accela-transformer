import google_benchmark as benchmark

import numpy as np
import torch

BATCH_SIZE = 2 ** 10
N = 1 ** 6

@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def batched_softmax_numpy(state):
    a = np.random.rand(BATCH_SIZE, N)
    while state:
      m = a.max(axis=1)
      res = np.exp(a - m)
      denom = res.sum(axis=1)
      res = res / denom


@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def batched_softmax_pytorch(state):
    a = torch.randn(BATCH_SIZE, N)
    while state:
        torch.nn.Softmax(a)


if __name__ == "__main__":
    benchmark.main()