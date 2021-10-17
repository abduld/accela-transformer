#!/usr/bin/env python3

### [import-package]
import math
import robopy as acc

### [import-package]

### [use-fast-exp]
from robopy._lang_python import fast_exp_mlas

### [use-fast-exp]

### [declare-input-length]
N = 2 ** 20
### [declare-input-length]

### [declare-target-depdendent-params]
target = acc.Target(category=acc.Target.Category.CPU)
vector_size = target.vector_bytes // 4
### [declare-target-depdendent-params]

### [declare-input-output-arrays]
Input = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
Output = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
### [declare-input-output-arrays]

### [declare-temp-arrays]
Denom = acc.Array(
    role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, shape=(1,)
)
MaxVal = acc.Array(
    role=acc.Array.Role.TEMP, element_type=acc.ScalarType.float32, shape=(1,)
)
### [declare-temp-arrays]

### [init]
init_nest = acc.Nest(shape=(1,))


@init_nest.iteration_logic
def _():
    MaxVal[0] = -math.inf
    Denom[0] = 0.0


init_schedule = init_nest.create_schedule()
### [init]

### [max]
max_nest = acc.Nest(shape=(N,))
m = max_nest.get_indices()


@max_nest.iteration_logic
def _():
    MaxVal[0] = acc.max(MaxVal[0], Input[m])


max_schedule = max_nest.create_schedule()
### [max]

### [exp]
exp_nest = acc.Nest(shape=(N,))
i = exp_nest.get_indices()


@exp_nest.iteration_logic
def _():
    Output[i] = fast_exp_mlas(Input[i] - MaxVal[0])


exp_schedule = exp_nest.create_schedule()
### [exp]

### [accum]
accum_nest = acc.Nest(shape=(N,))
a = accum_nest.get_indices()


@accum_nest.iteration_logic
def _():
    Denom[0] += Output[a]


accum_schedule = accum_nest.create_schedule()
### [accum]

### [divide]
div_nest = acc.Nest(shape=(N,))
j = div_nest.get_indices()


@div_nest.iteration_logic
def _():
    Output[j] /= Denom[0]


div_schedule = div_nest.create_schedule()
### [divide]

### [fuse]
fused_schedule1 = acc.fuse((init_schedule, max_schedule), partial=0)
fused_schedule2 = acc.fuse((exp_schedule, accum_schedule), partial=0)
fused_schedule3 = acc.fuse((fused_schedule1, fused_schedule2), partial=0)
fused_schedule = acc.fuse((fused_schedule3, div_schedule), partial=0)
### [fuse]

### [fuse-indices]
f, f3, f1, z, m, f2, i, a, j = fused_schedule.get_indices()
### [fuse-indices]

### [fused-schedule-split]
# mm = fused_schedule.split(m, 8 * vector_size)
ii = fused_schedule.split(i, 4 * vector_size)
# aa = fused_schedule.split(a, 4 * vector_size)
# jj = fused_schedule.split(j, 8 * vector_size)

mmm = fused_schedule.split(m, 2 * vector_size)
iii = fused_schedule.split(ii, 2 * vector_size)
aaa = fused_schedule.split(a, 2 * vector_size)
jjj = fused_schedule.split(j, 2 * vector_size)
### [fused-schedule-split]

### [fused-schedule-reorder]
fused_schedule.reorder(f, f3, f1, z, m, mmm, i, a, ii, f2, iii, aaa, j, jjj)
### [fused-schedule-reorder]

### [fused-plan]
fused_plan = fused_schedule.create_action_plan()
### [fused-plan]

### [fused-plan-unroll-vectorize]
# fused_plan.unroll(mm)
fused_plan.unroll(ii)
# fused_plan.unroll(aa)
# fused_plan.unroll(jj)

fused_plan.vectorize(mmm)
fused_plan.vectorize(iii)
fused_plan.unroll(aaa)  # TODO: Add vectorization
fused_plan.vectorize(jjj)
### [fused-plan-unroll-vectorize]

### [export-package]
package = acc.Package()
package.add_function(fused_plan, args=(Output, Input), base_name="vectorized")
package.build(
    name="vectorized",
    mode=acc.Package.Mode.RELEASE,
)
### [export-package]
