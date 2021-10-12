#!/usr/bin/env python3

import math
import robopy as rp

N = 2 ** 20
DEV_MODE = False

Input = rp.Array(
    role=rp.Array.Role.INPUT, element_type=rp.ScalarType.float32, shape=(N,)
)
Output = rp.Array(
    role=rp.Array.Role.INPUT_OUTPUT, element_type=rp.ScalarType.float32, shape=(N,)
)
Denom = rp.Array(
    role=rp.Array.Role.TEMP, element_type=rp.ScalarType.float32, shape=(1,)
)
MaxVal = rp.Array(
    role=rp.Array.Role.TEMP, element_type=rp.ScalarType.float32, shape=(1,)
)

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

div_nest = rp.Nest(shape=(N,))
j = div_nest.get_indices()


@div_nest.iteration_logic
def _():
    Output[j] /= Denom[0]


div_schedule = div_nest.create_schedule()

fused_schedule1 = rp.fuse((init_schedule, max_schedule), partial=0)
fused_schedule2 = rp.fuse((fused_schedule1, exp_schedule), partial=0)
fused_schedule3 = rp.fuse((fused_schedule2, accum_schedule), partial=0)
fused_schedule = rp.fuse((fused_schedule3, div_schedule), partial=0)

fused_plan = fused_schedule.create_action_plan()

target = rp.Target(category=rp.Target.Category.CPU)

package = rp.Package()
package.add_function(fused_plan, args=(Output, Input), base_name="naive")

package.build(
    name="naive",
    format=rp.Package.Format.HAT,
    mode=rp.Package.Mode.DEBUG if DEV_MODE else rp.Package.Mode.RELEASE,
)
