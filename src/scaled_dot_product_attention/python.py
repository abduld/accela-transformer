import google_benchmark as benchmark

import numpy as np
import torch

BATCH_SIZE = 2 ** 10
N = 1 ** 6

def row_softmax(input):
  m = input.max(axis=1)
  res = np.exp(input - m)
  denom = res.sum(axis=1)
  return res / denom
  
@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def scaled_dot_product_numpy(state):
    a = np.random.rand(BATCH_SIZE, N)
    while state:
        res = row_softmax(a) 


@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def scaled_dot_product_pytorch(state):
    a = torch.randn(BATCH_SIZE, N)
    while state:
        torch.nn.Softmax(a)


if __name__ == "__main__":
    benchmark.main()