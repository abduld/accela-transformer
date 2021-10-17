from typing import Sequence, NamedTuple
from robopy import Target, Array, Scalar, Nest, fuse

class Options(NamedTuple):
    ForceCacheBMatrix: bool = False
    BCacheSizeThreshold: int = 128**2
    KUnroll: int = 4
    NumRowsInKernel: int = 6
    NumColumnsInKernelScaleFactor: int = 2
    BMatrixTileSize: Sequence[int] = [128, 256]
    PackBFuncName: str = ""
    PackBBufferSizeFuncName: str = ""
    UseBiasFusion: bool = True
    

def Gemm(
        A: Array, B: Array, C: Array, Y: Array,
        transA=False, transB=False,
        alpha=1.0, beta=1.0, opts=Options(),
        target=Target.HOST):

    if not all(
        len(arr.shape) >= 2 and                 # check rank
        len(arr.shape) == len(A.shape) and      # rank's the same for all args
        arr.shape[:-2] == A.shape[:-2]          # stacks are the same sizes
        for arr in [A, B, Y]
    ):
        raise RuntimeError("Invalid shapes for arguments")

    def trans_no_trans(val: int, trans: bool):
        trans_val = -1 if val == -2 else -2
        return trans_val if trans else val

    stack = A.shape[:-2]
    _M_A, _K_A = A.shape[trans_no_trans(-2, transA)
                         ], A.shape[trans_no_trans(-1, transA)]
    _K_B, _N_B = B.shape[trans_no_trans(-2, transB)
                         ], B.shape[trans_no_trans(-1, transB)]
    _M_Y, _N_Y = A.shape[trans_no_trans(-2, transA)
                         ], B.shape[trans_no_trans(-1, transB)]

    if Y.shape != stack + [_M_Y, _N_Y]:
        raise RuntimeError("Incompatible shapes for arguments")

    if C:
        if len(C.shape) == len(Y.shape):
            if C.shape != Y.shape:
                raise RuntimeError("Incompatible shapes for arguments")

        elif len(C.shape) == (len(Y.shape) - 1):
            if C.shape != Y.shape[:-3] + Y.shape[-1:]:
                raise RuntimeError("Incompatible shapes for arguments")

        else:
            raise RuntimeError("Incompatible shapes for arguments")

    M = _M_Y
    N = _N_Y
    K = _K_A
    output_rows = M
    output_cols = N
    inner_dim = K

    # Schedule constants
    column_block = opts.BMatrixTileSize[1]
    inner_dim_block = opts.BMatrixTileSize[0]
    num_rows_in_kernel = opts.NumRowsInKernel
    num_cols_in_kernel = opts.NumColumnsInKernelScaleFactor * (target.vector_bytes // 4) # target.vector_bytes // 4 is how many 32-bit float elements can fit into the vector register

    # Apply a simple stretching to the kernel size to fit the output shape
    if num_cols_in_kernel > output_cols:
        while num_cols_in_kernel > output_cols:
            num_rows_in_kernel *= 2
            num_cols_in_kernel //= 2
    elif num_rows_in_kernel > output_rows:
        while num_rows_in_kernel > output_rows:
            num_rows_in_kernel //= 2
            num_cols_in_kernel *= 2

    # now clamp
    num_rows_in_kernel = int(min(num_rows_in_kernel, output_rows))
    num_cols_in_kernel = int(min(num_cols_in_kernel, output_cols))

    # Apply a simple stretching to the block sizes to use as much of
    # the original columnBlock x innerDimensionBlock area as possible
    while column_block > output_cols:
        if (column_block // 2) < num_cols_in_kernel:
            # Don't shrink the column block smaller than num_cols_in_kernel
            break
        column_block //= 2
        inner_dim_block *= 2
    while inner_dim_block > inner_dim:
        inner_dim_block //= 2
        column_block *= 2

    # Now clamp
    column_block = int(min(column_block, output_cols))
    inner_dim_block = int(min(inner_dim_block, inner_dim))

    bias_nest = Nest(shape=Y.shape)
    bias_idxs = bias_nest.get_indices()
    bias_stack_idxs, bias_i, bias_j = tuple(bias_idxs[:-2]), *bias_idxs[-2:]
    bias_y_idxs = bias_idxs
    bias_c_idxs = bias_stack_idxs + (bias_j,)

    if C is not None:
        @bias_nest.iteration_logic
        def _():
            Y[bias_y_idxs] = Scalar(beta) * C[bias_c_idxs]
    else:
        @bias_nest.iteration_logic
        def _():
            Y[bias_y_idxs] = Scalar(0.0)

    bias_schedule = bias_nest.create_schedule()

    compute_nest = Nest(shape=tuple(stack + [M, N, K]))
    compute_idxs = compute_nest.get_indices()
    compute_stack_idxs, compute_i, compute_j, compute_k = tuple(
        compute_idxs[:-3]), *compute_idxs[-3:]

    # This is only until we have proper caching introspection
    compute_Y_idxs = compute_stack_idxs + (compute_i, compute_j)
    compute_A_idxs = compute_stack_idxs + \
        (compute_i, compute_k) if not transA else (compute_k, compute_i)
    compute_B_idxs = compute_stack_idxs + \
        (compute_k, compute_j) if not transB else (compute_j, compute_k)

    @compute_nest.iteration_logic
    def _():
        Y[compute_Y_idxs] += Scalar(alpha) * \
            A[compute_A_idxs] * B[compute_B_idxs]

    compute_schedule = compute_nest.create_schedule()

    fused_schedule = fuse((bias_schedule, 
        compute_schedule), partial=len(stack) + 2)

    fused_idxs = fused_schedule.get_indices()
    f, fused_stack_idxs, fused_i, fused_j, k = fused_idxs[0], tuple(
        fused_idxs[1:-3]), *fused_idxs[-3:]

    fused_Y_idxs = fused_stack_idxs + (fused_i, fused_j)
    fused_A_idxs = fused_stack_idxs + \
        (fused_i, k) if not transA else (k, fused_i)
    fused_B_idxs = fused_stack_idxs + \
        (k, fused_j) if not transB else (fused_j, k)

    jj = fused_schedule.split(fused_j, column_block)
    kk = fused_schedule.split(k, inner_dim_block)
    kkk = fused_schedule.split(kk, opts.KUnroll)
    jjj = fused_schedule.split(jj, num_cols_in_kernel)
    jjjj = fused_schedule.split(jjj, target.vector_bytes // 4) # (target.vector_bytes // 4) is how many 32-bit float elements can fit into the vector register
    ii = fused_schedule.split(fused_i, num_rows_in_kernel)

    fused_schedule.reorder(*fused_stack_idxs, fused_j,
                           f, k, fused_i, jj, kk, kkk, ii, jjj, jjjj)
    fused_plan = fused_schedule.create_action_plan(target)

    if opts.PackBFuncName and opts.PackBBufferSizeFuncName:
        fused_plan.emit_runtime_init_pack(
            B, opts.PackBFuncName, opts.PackBBufferSizeFuncName)
    elif opts.ForceCacheBMatrix or (K * N) > opts.BCacheSizeThreshold:
        fused_plan.cache(B, jj)
    fused_plan.cache(Y, ii)

    fused_plan.unroll(jjj)
    fused_plan.unroll(ii)

    fused_plan.vectorize(jjjj)

    return fused_plan, (A, B, C, Y) if C else (A, B, Y)
