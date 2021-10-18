#!/usr/bin/env python3

import numpy as np

N = 2 ** 20

arry = np.full((N,), 1.0/N)
size = len(arry)

while size >= 2:
  tmp_size = size // 2
  tmp = np.zeros(tmp_size)
  for i in range(tmp_size):
    tmp[i] = arry[2*i] + arry[2*i+1]
  size = tmp_size
  arry = tmp

print(arry[0])