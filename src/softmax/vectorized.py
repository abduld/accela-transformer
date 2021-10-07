#!/usr/bin/env python3
import math
import robopy as rp

N = 2**20
DEV_MODE = False

Input = rp.Array(role=rp.Array.Role.INPUT, element_type=rp.ScalarType.float32, shape=(N,))
Output = rp.Array(role=rp.Array.Role.INPUT_OUTPUT, element_type=rp.ScalarType.float32, shape=(N,))
Denom = rp.Array(role=rp.Array.Role.TEMP, element_type=rp.ScalarType.float32, shape=(1,))
MaxVal = rp.Array(role=rp.Array.Role.TEMP, element_type=rp.ScalarType.float32, shape=(1,))

init_nest = rp.Nest(shape=(1,))
z = init_nest.get_indices()
@init_nest.iteration_logic
def _():
    MaxVal[0] = -math.inf 
    Denom[0] = 0.0
init_schedule = init_nest.create_schedule()

max_nest = rp.Nest(shape=(N,))
m = max_nest.get_indices()
@max_nest.iteration_logic
def _():
    MaxVal[0] = rp.max(MaxVal[0], Input[m])
max_schedule = max_nest.create_schedule()

exp_nest = rp.Nest(shape=(N,))
i = exp_nest.get_indices()
@exp_nest.iteration_logic
def _():
    Output[i] = rp.exp(Input[i] - MaxVal[0])
exp_schedule = exp_nest.create_schedule()

accum_nest = rp.Nest(shape=(N,))
a = accum_nest.get_indices()
@exp_nest.iteration_logic
def _():
    Denom[0] += Output[i]
accum_schedule = accum_nest.create_schedule()

div_nest = rp.Nest(shape=(N, ))
j = div_nest.get_indices()
@div_nest.iteration_logic
def _():
    Output[j] /= Denom[0]
div_schedule = div_nest.create_schedule()

fused_schedule1 = rp.fuse((init_schedule, max_schedule), partial=0)
fused_schedule2 = rp.fuse((fused_schedule1, exp_schedule), partial=0)
fused_schedule3 = rp.fuse((fused_schedule2, accum_schedule), partial=0)
fused_schedule = rp.fuse((fused_schedule3, div_schedule), partial=0)

f1, f2, f3, f4, z, m, i, a, j = fused_schedule.get_indices()


fused_plan = fused_schedule.create_action_plan()

target = rp.Target(category=rp.Target.Category.CPU)
tile_size = 128
zz = fused_schedule.split(z, tile_size)
mm = fused_schedule.split(m, tile_size)
ii = fused_schedule.split(i, tile_size)
aa = fused_schedule.split(a, tile_size)
jj = fused_schedule.split(j, tile_size)

zzz = fused_schedule.split(zz, 2 * target.vector_bytes // 4)
mmm = fused_schedule.split(mm, 2 * target.vector_bytes // 4)
iii = fused_schedule.split(ii, 2 * target.vector_bytes // 4)
aaa = fused_schedule.split(aa, 2 * target.vector_bytes // 4)
jjj = fused_schedule.split(jj, 2 * target.vector_bytes // 4)
fused_schedule.reorder(f1, f2, f3, f4, z, zz, zzz, m, mm, mmm, i, ii, iii, a, aa, aaa, j, jj, jjj)

fused_plan = fused_schedule.create_action_plan()

# fused_plan.unroll(zz)
# fused_plan.unroll(mm) 
# fused_plan.unroll(ii) 
# fused_plan.unroll(aa) 
# fused_plan.unroll(jj)

fused_plan.vectorize(zzz) 
fused_plan.vectorize(mmm) 
fused_plan.vectorize(iii) 
fused_plan.vectorize(aaa) 
fused_plan.vectorize(jjj)


package = rp.Package()
package.add_function(fused_plan, args=(Output, Input), base_name="vectorized")
package.build(name="vectorized", format=rp.Package.Format.HAT)
