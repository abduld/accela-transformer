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


np.random.seed(0)
torch.manual_seed(0)

Q = np.random.randn(SEQUENCE_LENGTH,DK)
K = np.random.randn(SEQUENCE_LENGTH,DK)
V = np.random.randn(SEQUENCE_LENGTH,DK)

def row_softmax(a):
  m = a.max(axis=-1)  
  res = np.exp(a - np.vstack(m))
  denom = res.sum(axis=-1) 
  res = res / np.vstack(denom)
  return res
  
@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def scaled_dot_product_numpy(state):
    q = Q
    k = K
    v = V
    def dropout(x): 
      u1 = np.random.binomial(1,1-ATTN_DROPOUT,size=x.shape)/(1-ATTN_DROPOUT)
      return x  
    output = None
    while state: 
        attn = np.dot(q / TEMP, k.transpose(1,0))
        x = attn
        attn = dropout(row_softmax(attn))
        output = np.dot(attn, v) 
        output = x
    # print(output.shape)
    # print(output[0,:10])


@benchmark.register
@benchmark.option.unit(benchmark.kMicrosecond)
def scaled_dot_product_pytorch(state):

    q = torch.from_numpy(Q)
    k = torch.from_numpy(K)
    v = torch.from_numpy(V)
    
    dropout = nn.Dropout(p=ATTN_DROPOUT) 
    output = None
    while state:   
        attn = torch.matmul(q / TEMP, k.transpose(1,0))
        x = attn
        attn = nn.functional.softmax(attn,dim=-1)
        output = torch.matmul(attn, v) 
        output = x
    # print(output.shape)
    # print(output[0,:10])
        


if __name__ == "__main__":
    benchmark.main()