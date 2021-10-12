#!/usr/bin/env python3
import math
import robopy as acc
from robopy._lang_python import fast_exp

N = 2 ** 10
DEV_MODE = False

target = acc.Target(category=acc.Target.Category.CPU)

Input = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
Output = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
Denom = acc.Array(
    role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, shape=(1,)
)
MaxVal = acc.Array(
    role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, shape=(1,)
)

init_nest = acc.Nest(shape=(1,))
z = init_nest.get_indices()


@init_nest.iteration_logic
def _():
    MaxVal[0] = -math.inf
    Denom[0] = 0.0


init_schedule = init_nest.create_schedule()

max_nest = acc.Nest(shape=(N,))
m = max_nest.get_indices()


@max_nest.iteration_logic
def _():
    MaxVal[0] = acc.max(MaxVal[0], Input[m])


max_schedule = max_nest.create_schedule()

exp_nest = acc.Nest(shape=(N,))
i = exp_nest.get_indices()


@exp_nest.iteration_logic
def _():
    Output[i] = fast_exp(Input[i] - MaxVal[0])
    # Output[i] = rp.exp(Input[i] - MaxVal[0])


exp_schedule = exp_nest.create_schedule()

accum_nest = acc.Nest(shape=(N,))
a = accum_nest.get_indices()


@accum_nest.iteration_logic
def _():
    Denom[0] += Output[a]


accum_schedule = accum_nest.create_schedule()

div_nest = acc.Nest(shape=(N,))
j = div_nest.get_indices()


@div_nest.iteration_logic
def _():
    Output[j] /= Denom[0]


div_schedule = div_nest.create_schedule()

fused_schedule1 = acc.fuse((init_schedule, max_schedule), partial=0)
fused_schedule2 = acc.fuse((exp_schedule, accum_schedule), partial=0)
fused_schedule3 = acc.fuse((fused_schedule1, fused_schedule2), partial=0)
fused_schedule = acc.fuse((fused_schedule3, div_schedule), partial=0)

f, f3, f1, z, m, f2, i, a, j = fused_schedule.get_indices()

# print((f, f3, f1, f2, z, m, i, j))
# print(fused_schedule._fusing_index)
# print(fused_schedule2._fusing_index)
# f1, z, f2, m, f3, i, f4, a, j = fused_schedule.get_indices()
# f1, f2, f3, f4, z, m, i, a, j = fused_schedule.get_indices()


fused_plan = fused_schedule.create_action_plan()

# tile_size = 8 * target.vector_bytes // 4
# zz = fused_schedule.split(z, tile_size)
# mm = fused_schedule.split(m, tile_size)
# ii = fused_schedule.split(i, tile_size)
# aa = fused_schedule.split(a, tile_size)
# jj = fused_schedule.split(j, tile_size)

# zzz = fused_schedule.split(z, 4 * target.vector_bytes // 4)
# mmm = fused_schedule.split(m, 4 * target.vector_bytes // 4)
# iii = fused_schedule.split(i, 4 * target.vector_bytes // 4)
# jjj = fused_schedule.split(j, 4 * target.vector_bytes // 4)

zzzz = fused_schedule.split(z, 2 * target.vector_bytes // 4)
mmmm = fused_schedule.split(m, 2 * target.vector_bytes // 4)
iiii = fused_schedule.split(i, 2 * target.vector_bytes // 4)
aaaa = fused_schedule.split(a, 2 * target.vector_bytes // 4)
jjjj = fused_schedule.split(j, 2 * target.vector_bytes // 4)

# fused_schedule.reorder(
#     f4, z, zz, f3, m, mm, f2, i, ii, f1, a, aa, j, jj
# )
# fused_schedule.reorder(
#     f1, f2, f3, f4, z, zz, zzz, zzzz, m, mm, mmm, mmmm, i, ii, iii, iiii, a, aa, aaa, aaaa, j, jj, jjj, jjjj
# )
# print((f1, f2, f3, f4))
# print((z, zz, zzz, m, mm, mmm, i, ii, iii, a, aa, aaa, j, jj, jjj))
# fused_schedule.reorder(
#     f1, z, zz, zzz, f2, m, mm, mmm, f3, i, ii, iii, f4, a, aa, aaa, j, jj, jjj
# )
# fused_schedule.reorder(
#     f1, f2, f3, f4, z, zz, zzz, m, mm, mmm, i, ii, iii, a, aa, aaa, j, jj, jjj
# )
# fused_schedule.reorder(f1, f2, f3, f4, z, zz, m, mm, i, ii, a, aa, j, jj)

fused_schedule.reorder(f, f3, f1, z, zzzz, m, mmmm, f2, i, iiii, a, aaaa, j, jjjj)

# fused_plan = fused_schedule.create_action_plan()

# fused_plan.unroll(zzz)
# fused_plan.unroll(mmm)
# fused_plan.unroll(iii)
# fused_plan.unroll(jjj)

fused_plan.vectorize(zzzz)
fused_plan.vectorize(mmmm)
fused_plan.vectorize(iiii)
fused_plan.unroll(aaaa)
fused_plan.vectorize(jjjj)


package = acc.Package()
package.add_function(fused_plan, args=(Output, Input), base_name="vectorized")
package.build(
    name="vectorized",
    mode=acc.Package.Mode.DEBUG if DEV_MODE else acc.Package.Mode.RELEASE,
)
