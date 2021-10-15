
import random
import time


import google_benchmark as benchmark
from google_benchmark import Counter

import numpy as np

BATCH_SIZE = 2 ** 10
N = 1 ** 6

def row_softmax(input):
  m = input.max(axis=1)
  res = np.exp(input - m)
  denom = res.sum(axis=1)
  return res / denom
 
@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def scaled_dot_product(state):
    a = np.random.rand(BATCH_SIZE, N)
    while state:
        res = row_softmax(a) 


if __name__ == "__main__":
    benchmark.main()