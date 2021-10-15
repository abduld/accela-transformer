import google_benchmark as benchmark
import math
import numpy as np
import torch
from torch import nn


BATCH_SIZE = 1
SEQUENCE_LENGTH = 10
DM =  768
DFF = 3072
DK = 64
DV = 64
NUM_HEADS = 12

DK = 64
TEMP = DK ** 0.5
ATTN_DROPOUT = 0.1

def row_softmax(input):
  m = input.max(axis=1)
  res = np.exp(input - m)
  denom = res.sum(axis=1)
  return res / denom
  
@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def scaled_dot_product_numpy(state):
    np.random.seed(0)
    q = np.random.randn(SEQUENCE_LENGTH,DK)
    k = np.random.randn(SEQUENCE_LENGTH,DK)
    v = np.random.randn(SEQUENCE_LENGTH,DK)
    def dropout(x): 
      u1 = np.random.binomial(1,ATTN_DROPOUT,size=x.shape)/ATTN_DROPOUT
      return x * u1
    output = None
    while state: 
        attn = np.dot(q / TEMP, k.transpose(1,0))
        attn = dropout(row_softmax(attn))
        output = np.dot(attn, v) 
    # print(output.shape)
    # print(output[0,:10])


@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def scaled_dot_product_pytorch(state):
    np.random.seed(0)
    torch.manual_seed(0)
    q = torch.randn(SEQUENCE_LENGTH,DK)
    k = torch.randn(SEQUENCE_LENGTH,DK)
    v = torch.randn(SEQUENCE_LENGTH,DK)
    dropout = nn.Dropout(p=ATTN_DROPOUT) 
    output = None
    while state:   
        attn = torch.matmul(q / TEMP, k.transpose(1,0))
        attn = dropout(nn.functional.softmax(attn,dim=1))
        output = torch.matmul(attn, v) 
    # print(output.shape)
    # print(output[0,:10])
        


if __name__ == "__main__":
    benchmark.main()