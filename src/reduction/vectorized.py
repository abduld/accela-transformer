#!/usr/bin/env python3

import math
import robopy as acc

N = 2 ** 20
DEV_MODE = False

target = acc.Target(category=acc.Target.Category.CPU)
vector_size = (
    target.vector_bytes // 4
)  # AVX-2 gives 256-bit registers, which can hold 8 floats
vector_units = 2 * vector_size  # AVX-2 has 16 256-bit registers

Input = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
SumVec = acc.Array(
    role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, shape=(vector_size,)
) 
Sum = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(1,)
) 
 
vec_accum_nest = acc.Nest(shape=(N // vector_size, vector_size))
i, j = vec_accum_nest.get_indices()

@vec_accum_nest.iteration_logic
def _():
    SumVec[j] += Input[i * vector_size + j]

 
finalize_accum_nest = acc.Nest(shape=(vector_size,))
k = finalize_accum_nest.get_indices()
@finalize_accum_nest.iteration_logic
def _():
    Sum[0] += SumVec[k]

vec_accum_schedule = vec_accum_nest.create_schedule()
finalize_accum_schedule = finalize_accum_nest.create_schedule()

fused_schedule = acc.fuse((vec_accum_schedule, finalize_accum_schedule), partial=0)

fused_plan = fused_schedule.create_action_plan() 

f, i, j, k = fused_schedule.get_indices()

fused_plan.vectorize(j)


package = acc.Package()
package.add_function(fused_plan, args=(Sum, Input), base_name="vectorized")

package.build(
    name="vectorized",
    format=acc.Package.Format.HAT,
    mode=acc.Package.Mode.DEBUG if DEV_MODE else acc.Package.Mode.RELEASE,
)
