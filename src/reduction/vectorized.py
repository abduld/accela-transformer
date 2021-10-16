#!/usr/bin/env python3

### [import-package]
import robopy as acc
### [import-package]
 

### [declare-target-dependent-properties]
target = acc.Target(category=acc.Target.Category.CPU)
vector_size = (
    target.vector_bytes // 4
)  # AVX-2 gives 256-bit registers, which can hold 8 floats
vector_units = 2 * vector_size  # AVX-2 has 16 256-bit registers
split_size = 4 * vector_size
### [declare-target-dependent-properties]
 
N = 2 ** 20 

### [declare-inputs]
Input = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
Sum = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(1,)
)
### [declare-inputs]

### [declare-input-vec]
SumVec = acc.Array(
    role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, shape=(split_size,)
)
### [declare-input-vec]


### [declare-vector-reduction-iteration-logic]
vec_accum_nest = acc.Nest(shape=(N // split_size, split_size))
i, j = vec_accum_nest.get_indices()

@vec_accum_nest.iteration_logic
def _():
    SumVec[j] += Input[i * split_size + j]
### [declare-vector-reduction-iteration-logic]


### [declare-horizontal-reduction-iteration-logic]
finalize_accum_nest = acc.Nest(shape=(split_size,))
k = finalize_accum_nest.get_indices()

@finalize_accum_nest.iteration_logic
def _():
    Sum[0] += SumVec[k]
### [declare-horizontal-reduction-iteration-logic]


### [create-two-schedules]
vec_accum_schedule = vec_accum_nest.create_schedule()
finalize_accum_schedule = finalize_accum_nest.create_schedule()
### [create-two-schedules]

### [fuse-two-schedules]
fused_schedule = acc.fuse((vec_accum_schedule, finalize_accum_schedule), partial=0)
### [fuse-two-schedules]

### [get-fused-schedule-indices]
f, i, j, k = fused_schedule.get_indices()
### [get-fused-schedule-indices]

### [split-index-by-vector-units]
ii = fused_schedule.split(i, vector_units)
### [split-index-by-vector-units]


### [create-fused-action-plan]
fused_plan = fused_schedule.create_action_plan()
### [create-fused-action-plan]

### [optimize-indices]
fused_plan.unroll(ii)
fused_plan.vectorize(j)
fused_plan.unroll(k) 
### [optimize-indices]


### [create-package]
package = acc.Package()
package.add_function(fused_plan, args=(Sum, Input, SumVec), base_name="vectorized")
### [create-package]

### [export-package]
package.build(
    name="vectorized",
    format=acc.Package.Format.HAT,
    mode=acc.Package.Mode.RELEASE,
)
### [export-package]
