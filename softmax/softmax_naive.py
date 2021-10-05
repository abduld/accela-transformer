#!/usr/bin/env python3

import robopy as rp

N = 2**20

Input = rp.Array(role=rp.Array.Role.INPUT, element_type=rp.ScalarType.float32, shape=(N,))
Output = rp.Array(role=rp.Array.Role.INPUT_OUTPUT, element_type=rp.ScalarType.float32, shape=(N,))
Denom = rp.Array(role=rp.Array.Role.TEMP, element_type=rp.ScalarType.float32, shape=(1,))

exp_nest = rp.Nest(shape=(N,))

i = exp_nest.get_indices()

@exp_nest.iteration_logic
def _():
    Output[i] = rp.exp(Input[i])
    Denom[0] += Output[i]

exp_schedule = exp_nest.create_schedule()

finalize_nest = rp.Nest(shape=(N, ))
j = finalize_nest.get_indices()

@finalize_nest.iteration_logic
def _():
    Output[j] = Output[j] / Denom[0]


final_schedule = finalize_nest.create_schedule()
fused_schedule = rp.fuse((exp_schedule, final_schedule), partial=0)
f, i, j = fused_schedule.get_indices()


fused_plan = fused_schedule.create_action_plan()

target = rp.Target(category=rp.Target.Category.CPU)

package = rp.Package()
package.add_function(fused_plan, args=(Output, Input), base_name="softmax_naive")

package.build(name="softmax_naive", format=rp.Package.Format.HAT)
