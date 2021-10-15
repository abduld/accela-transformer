import google_benchmark as benchmark

import numpy as np
import torch

N = 2 ** 20

np.random.seed(0)
torch.manual_seed(0)
input = np.random.rand(N)

@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def softmax_numpy(state):
    a = input 
    res = None
    while state:
      m = a.max()
      res = np.exp(a - m)
      denom = res.sum() 
      res = res / denom
    # print(res[:10])


@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def softmax_pytorch(state):
    a = torch.from_numpy(input) 
    res = None
    while state:
        res = torch.nn.functional.softmax(a,dim=-1)
    # print(res[:10])


if __name__ == "__main__":
    benchmark.main()