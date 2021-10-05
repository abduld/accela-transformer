#!/usr/bin/env python3

import math
import robopy as rp

N = 2**20

Input = rp.Array(role=rp.Array.Role.INPUT, element_type=rp.ScalarType.float32, shape=(N,))
Output = rp.Array(role=rp.Array.Role.INPUT_OUTPUT, element_type=rp.ScalarType.float32, shape=(N,))
Denom = rp.Array(role=rp.Array.Role.TEMP, element_type=rp.ScalarType.float32, shape=(1,))
MaxVal = rp.Array(role=rp.Array.Role.TEMP, element_type=rp.ScalarType.float32, shape=(1,))

zero_nest = rp.Nest(shape=(1,))
z = zero_nest.get_indices()
@zero_nest.iteration_logic
def _():
    MaxVal[z] = -math.inf
    Denom[z] = 0.0
zero_schedule = zero_nest.create_schedule()

max_nest = rp.Nest(shape=(N,))
i = max_nest.get_indices()
@max_nest.iteration_logic
def _():
    MaxVal[i] = rp.max(MaxVal[i], Input[i])
max_schedule = max_nest.create_schedule()

exp_nest = rp.Nest(shape=(N,))
i = exp_nest.get_indices()
@exp_nest.iteration_logic
def _():
    Output[i] = rp.exp(Input[i] - MaxVal[0])
    Denom[0] += Output[i]
exp_schedule = exp_nest.create_schedule()

finalize_nest = rp.Nest(shape=(N, ))
j = finalize_nest.get_indices()
@finalize_nest.iteration_logic
def _():
    Output[j] = Output[j] / Denom[0]
final_schedule = finalize_nest.create_schedule()

fused_schedule = rp.fuse((zero_schedule, max_schedule, exp_schedule, final_schedule), partial=0)
f, z, i, j, k = fused_schedule.get_indices()


fused_plan = fused_schedule.create_action_plan()

target = rp.Target(category=rp.Target.Category.CPU)

package = rp.Package()
package.add_function(fused_plan, args=(Output, Input), base_name="softmax_naive")

package.build(name="softmax_naive", format=rp.Package.Format.HAT)
