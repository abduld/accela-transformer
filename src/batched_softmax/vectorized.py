#!/usr/bin/env python3
import math
import robopy as acc
from robopy._lang_python import fast_exp_mlas

BATCH_SIZE = 2 ** 10
N = 2 ** 6
DEV_MODE = False

target = acc.Target(category=acc.Target.Category.CPU)
vector_size = target.vector_bytes // 4 # AVX-2 gives 256-bit registers, which can hold 8 floats
vector_units = 2 * vector_size # AVX-2 has 16 256-bit registers
vector_ports = 8 * vector_units

package = acc.Package()

Input = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(BATCH_SIZE, N)
)
Output = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(BATCH_SIZE, N)
)
Denom = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(BATCH_SIZE,)
)
MaxVal = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(BATCH_SIZE,)
)


def max():

    max_nest = acc.Nest(shape=(BATCH_SIZE, N))
    bm, m = max_nest.get_indices()

    @max_nest.iteration_logic
    def _():
        MaxVal[bm] = acc.max(MaxVal[bm], Input[bm, m])

    max_schedule = max_nest.create_schedule()
    bm, m = max_schedule.get_indices()
    mm = max_schedule.split(m, vector_ports)
    mmm = max_schedule.split(mm, vector_units)

    max_schedule.reorder(bm, m, mm, mmm)

    max_plan = max_schedule.create_action_plan()

    max_plan.unroll(mm)
    max_plan.vectorize(mmm)

    package.add_function(max_plan, args=(MaxVal, Input), base_name="vectorized_max")


def exp():
    exp_nest = acc.Nest(shape=(BATCH_SIZE, N))
    bi, i = exp_nest.get_indices()


    @exp_nest.iteration_logic
    def _():
        Output[bi, i] = fast_exp_mlas(Input[bi, i] - MaxVal[bi])

    exp_schedule = exp_nest.create_schedule()
    bi, i = exp_schedule.get_indices()
    ii = exp_schedule.split(i, vector_ports)
    iii = exp_schedule.split(ii, vector_units)

    exp_schedule.reorder(bi, i, ii, iii)

    exp_plan = exp_schedule.create_action_plan()

    exp_plan.unroll(ii)
    exp_plan.vectorize(iii)

    package.add_function(
        exp_plan, args=(Output, Input, MaxVal), base_name="vectorized_exp"
    )


def accum():
    accum_nest = acc.Nest(shape=(BATCH_SIZE, N))
    ba, a = accum_nest.get_indices()

    @accum_nest.iteration_logic
    def _():
        Denom[ba] += Output[ba, a]

    accum_schedule = accum_nest.create_schedule()
    ba, a = accum_schedule.get_indices()
    baa = accum_schedule.split(ba, vector_ports) 
    baaa = accum_schedule.split(baa, vector_ports) 

    accum_schedule.reorder(ba, a, baa, baaa)

    accum_plan = accum_schedule.create_action_plan()

    accum_plan.unroll(baa)
    accum_plan.vectorize(baaa)

    package.add_function(
        accum_plan, args=(Denom, Output), base_name="vectorized_accum"
    )


def div():
    div_nest = acc.Nest(shape=(BATCH_SIZE, N))
    bj, j = div_nest.get_indices()

    @div_nest.iteration_logic
    def _():
        Output[bj, j] /= Denom[bj]

    div_schedule = div_nest.create_schedule()
    bj, j = div_schedule.get_indices()
    jj = div_schedule.split(j, vector_ports)
    jjj = div_schedule.split(jj, vector_units)

    div_schedule.reorder(bj, j, jj, jjj)

    div_plan = div_schedule.create_action_plan()

    div_plan.unroll(jj)
    div_plan.vectorize(jjj)

    package.add_function(div_plan, args=(Denom, Output), base_name="vectorized_div")


max()
exp()
accum()
div()

package.build(
    name="vectorized",
    mode=acc.Package.Mode.DEBUG if DEV_MODE else acc.Package.Mode.RELEASE,
)
