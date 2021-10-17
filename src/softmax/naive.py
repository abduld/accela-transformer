#!/usr/bin/env python3

import math
import robopy as acc

N = 2 ** 20

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
    Output[i] = acc.exp(Input[i] - MaxVal[0])


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
fused_schedule2 = acc.fuse((fused_schedule1, exp_schedule), partial=0)
fused_schedule3 = acc.fuse((fused_schedule2, accum_schedule), partial=0)
fused_schedule = acc.fuse((fused_schedule3, div_schedule), partial=0)

fused_plan = fused_schedule.create_action_plan()

target = acc.Target(category=acc.Target.Category.CPU)

package = acc.Package()
package.add_function(fused_plan, args=(Output, Input), base_name="naive")

package.build(
    name="naive",
    format=acc.Package.Format.HAT,
    mode=acc.Package.Mode.RELEASE,
)
