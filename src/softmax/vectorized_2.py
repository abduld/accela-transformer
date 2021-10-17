#!/usr/bin/env python3

### [import-package]
import math
import robopy as acc
from robopy._lang_python import fast_exp_mlas
### [import-package]

### [declare-input-length]
N = 2 ** 20
### [declare-input-length]

### [declare-target]
target = acc.Target(category=acc.Target.Category.CPU)
vector_size = target.vector_bytes // 4 # AVX-2 gives 256-bit registers, which can hold 8 floats
vector_units = 2 * vector_size # AVX-2 has 16 256-bit registers
vector_ports = 8 * vector_units
### [declare-target]

### [declare-package]
package = acc.Package()
### [declare-package]

### [declare-arrays]
Input = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
Output = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
Denom = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(1,)
)
MaxVal = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(1,)
)
### [declare-arrays]


### [max]
def max():
    max_nest = acc.Nest(shape=(N,))
    m = max_nest.get_indices()

    @max_nest.iteration_logic
    def _():
        MaxVal[0] = acc.max(MaxVal[0], Input[m])

    max_schedule = max_nest.create_schedule()
    (m,) = max_schedule.get_indices()
    mm = max_schedule.split(m, vector_ports)
    mmm = max_schedule.split(mm, vector_units)

    max_plan = max_schedule.create_action_plan()

    max_plan.unroll(mm)
    max_plan.vectorize(mmm)

    package.add_function(max_plan, args=(MaxVal, Input), base_name="vectorized_2_max")
### [max]

### [exp]
def exp():
    exp_nest = acc.Nest(shape=(N,))
    i = exp_nest.get_indices()

    @exp_nest.iteration_logic
    def _():
        Output[i] = fast_exp_mlas(Input[i] - MaxVal[0])

    exp_schedule = exp_nest.create_schedule()
    (i,) = exp_schedule.get_indices()
    ii = exp_schedule.split(i, vector_ports)
    iii = exp_schedule.split(ii, vector_units)

    exp_plan = exp_schedule.create_action_plan()

    exp_plan.unroll(ii)
    exp_plan.vectorize(iii)

    package.add_function(
        exp_plan, args=(Output, Input, MaxVal), base_name="vectorized_2_exp"
    )
### [exp]


### [accum]
def accum():
    accum_nest = acc.Nest(shape=(N,))
    a = accum_nest.get_indices()

    @accum_nest.iteration_logic
    def _():
        Denom[0] += Output[a]

    accum_schedule = accum_nest.create_schedule()
    (a,) = accum_schedule.get_indices()
    aa = accum_schedule.split(a, vector_ports) 

    accum_plan = accum_schedule.create_action_plan()

    accum_plan.unroll(aa) 

    package.add_function(
        accum_plan, args=(Denom, Output), base_name="vectorized_2_accum"
    )
### [accum]


### [div]
def div():
    div_nest = acc.Nest(shape=(N,))
    j = div_nest.get_indices()

    @div_nest.iteration_logic
    def _():
        Output[j] /= Denom[0]

    div_schedule = div_nest.create_schedule()
    (j,) = div_schedule.get_indices()
    jj = div_schedule.split(j, vector_ports)
    jjj = div_schedule.split(jj, vector_units)

    div_plan = div_schedule.create_action_plan()

    div_plan.unroll(jj)
    div_plan.vectorize(jjj)

    package.add_function(div_plan, args=(Denom, Output), base_name="vectorized_2_div")
### [div]

### [add-to-package]
max()
exp()
accum()
div()
### [add-to-package]


### [export-package]
package.build(
    name="vectorized_2",
    mode=acc.Package.Mode.DEBUG if DEV_MODE else acc.Package.Mode.RELEASE,
)
### [export-package]
