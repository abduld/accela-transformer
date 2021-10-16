#!/usr/bin/env python3

import math
import robopy as acc

N = 2 ** 20
DEV_MODE = False

Input = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
Sum = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(1,)
) 
 
accum_nest = acc.Nest(shape=(N,))
i = accum_nest.get_indices()


@accum_nest.iteration_logic
def _():
    Sum[0] += Input[i]

accum_schedule = accum_nest.create_schedule()
 
plan = accum_schedule.create_action_plan()

target = acc.Target(category=acc.Target.Category.CPU)

package = acc.Package()
package.add_function(plan, args=(Sum, Input), base_name="naive")

package.build(
    name="naive",
    format=acc.Package.Format.HAT,
    mode=acc.Package.Mode.DEBUG if DEV_MODE else acc.Package.Mode.RELEASE,
)
