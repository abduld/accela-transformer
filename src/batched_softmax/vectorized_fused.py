#!/usr/bin/env python3

### [naive]
import math
import robopy as acc 

BATCH_SIZE = 2 ** 10
N = 2 ** 6 

Input = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(BATCH_SIZE, N)
)
Output = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(BATCH_SIZE, N)
)
Denom = acc.Array(
    role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, shape=(BATCH_SIZE,)
)
MaxVal = acc.Array(
    role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, shape=(BATCH_SIZE,)
)

init_nest = acc.Nest(shape=(BATCH_SIZE,))
z = init_nest.get_indices()

@init_nest.iteration_logic
def _():
    MaxVal[z] = -math.inf
    Denom[z] = 0.0

init_schedule = init_nest.create_schedule()

max_nest = acc.Nest(shape=(BATCH_SIZE, N))
bm, m = max_nest.get_indices()

@max_nest.iteration_logic
def _():
    MaxVal[bm] = acc.max(MaxVal[bm], Input[bm, m])

max_schedule = max_nest.create_schedule()

exp_nest = acc.Nest(shape=(BATCH_SIZE, N))
bi, i = exp_nest.get_indices()

@exp_nest.iteration_logic
def _():
    Output[bi, i] = acc.exp(Input[bi, i] - MaxVal[bi])

exp_schedule = exp_nest.create_schedule()

accum_nest = acc.Nest(shape=(BATCH_SIZE, N))
ba, a = accum_nest.get_indices()

@accum_nest.iteration_logic
def _():
    Denom[ba] += Output[ba, a]

accum_schedule = accum_nest.create_schedule()

div_nest = acc.Nest(shape=(BATCH_SIZE, N))
bj, j = div_nest.get_indices()

@div_nest.iteration_logic
def _():
    Output[bj, j] /= Denom[bj]

div_schedule = div_nest.create_schedule()
### [naive]

### [fuse]
fused_schedule1 = acc.fuse((init_schedule, max_schedule), partial=0)
fused_schedule2 = acc.fuse((fused_schedule1, exp_schedule), partial=0)
fused_schedule3 = acc.fuse((fused_schedule2, accum_schedule), partial=0)
fused_schedule = acc.fuse((fused_schedule3, div_schedule), partial=0)
### [fuse]


### [schedule]
### [schedule]

### [vectorize]
fused_plan = fused_schedule.create_action_plan()
### [vectorize]



### [export-package]
target = acc.Target(category=acc.Target.Category.CPU)

package = acc.Package()
package.add_function(fused_plan, args=(Output, Input), base_name="naive")

package.build(
    name="naive",
    format=acc.Package.Format.HAT,
    mode=acc.Package.Mode.RELEASE,
)
### [export-package]