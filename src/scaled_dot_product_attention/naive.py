#!/usr/bin/env python3
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import math
import robopy as acc
from matmul import Gemm
from batched_softmax.vectorized import softmax

DEV_MODE = False


BATCH_SIZE = 1
SEQUENCE_LENGTH = 10
DM =  768
DFF = 3072
DK = 64
DV = 64
NUM_HEADS = 12

DK = 64
TEMPERATURE = DK ** 0.5
TEMPERATURE_INV = 1.0 / TEMPERATURE
ATTN_DROPOUT = 0.1

target = acc.Target(category=acc.Target.Category.CPU)

Q = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(SEQUENCE_LENGTH,DM))
K = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(SEQUENCE_LENGTH,DM))
V = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(SEQUENCE_LENGTH,DM))


QK = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(SEQUENCE_LENGTH,SEQUENCE_LENGTH))

Output = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(SEQUENCE_LENGTH,DM))


package = acc.Package()

gemm_qk_plan, _ = Gemm(Q, K, None, QK, transB=True, alpha=TEMPERATURE_INV, beta=0.0, target=target)
softmax_plan, softmax_args = softmax(package, QK, QK, base_name="naive_softmax")
gemm_qkv_plan, _ = Gemm(QK, V, None, Output, beta=0.0, target=target)

_, _, MaxElems, Denoms = softmax_args

gemm_qk_fn = package.add_function(gemm_qk_plan, args=(Q, K, QK), base_name="naive_gemm_qk")
softmax_fn = package.add_function(softmax_plan, args=softmax_args, base_name="naive_softmax")
gemm_qkv_fn = package.add_function(gemm_qkv_plan, args=(QK, V, Output), base_name="naive_gemm_qkv")
 
def scaled_dot_product_attention(Q, K, V, Output, QK, MaxElems, Denoms):
  gemm_qk_fn(Q, K, QK)
  softmax_fn(QK, QK, MaxElems, Denoms)
  gemm_qkv_fn(QK, V, Output)

scaled_dot_product_attention_fn = package.add_function(scaled_dot_product_attention, args=(Q, K, V, Output, QK, MaxElems, Denoms), base_name="naive_scaled_dot_product_attention")



package.build(
    name="naive",
    format=acc.Package.Format.HAT,
    mode=acc.Package.Mode.DEBUG if DEV_MODE else acc.Package.Mode.RELEASE,
)
