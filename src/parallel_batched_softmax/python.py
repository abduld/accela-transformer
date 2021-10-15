import google_benchmark as benchmark

import numpy as np
import torch

BATCH_SIZE = 2 ** 10
N = 2 ** 6

np.random.seed(0)
torch.manual_seed(0)
input = np.random.rand(BATCH_SIZE, N)
 
@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def batched_softmax_numpy(state):
    a = input
    res = None
    while state:
      m = a.max(axis=-1)  
      res = np.exp(a - np.vstack(m))
      denom = res.sum(axis=-1) 
      res = res / np.vstack(denom)
    # print(res[0,:10])
    # print(res.shape)


@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def batched_softmax_pytorch(state):
    a = torch.from_numpy(input) 
    # print(input.shape)
    res = None
    while state:
        res = torch.nn.functional.softmax(a,dim=-1)
    # print(res[0,:10])
    # print(res.shape)


if __name__ == "__main__":
    benchmark.main()