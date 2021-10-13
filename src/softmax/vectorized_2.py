#!/usr/bin/env python3
import math
import robopy as acc
from robopy._lang_python import fast_exp

N = 2 ** 10
DEV_MODE = False

target = acc.Target(category=acc.Target.Category.CPU)
vector_size = target.vector_bytes // 4

package = acc.Package()

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

def max():
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


  max_schedule = acc.fuse((init_schedule, max_schedule), partial=0)
  f, z, m = max_schedule.get_indices()
  mm = max_schedule.split(m, 8 * vector_size)
  mmm = max_schedule.split(mm, 2 * vector_size)

  max_plan = max_schedule.create_action_plan()

  max_plan.unroll(mm)
  max_plan.vectorize(mmm)

  package.add_function(max_plan, args=(MaxVal, Denom, Input), base_name="vectorized_2_max")

def exp():
  exp_nest = acc.Nest(shape=(N,))
  i = exp_nest.get_indices()


  @exp_nest.iteration_logic
  def _():
      Output[i] = fast_exp(Input[i] - MaxVal[0]) 


  exp_schedule = exp_nest.create_schedule()
  i, = exp_schedule.get_indices()
  ii = exp_schedule.split(i, 8 * vector_size)
  iii = exp_schedule.split(ii, 2 * vector_size)

  exp_plan = exp_schedule.create_action_plan()

  exp_plan.unroll(ii)
  exp_plan.vectorize(iii)

  package.add_function(exp_plan, args=(Output, Input, MaxVal), base_name="vectorized_2_exp")

def accum():
  accum_nest = acc.Nest(shape=(N,))
  a = accum_nest.get_indices()


  @accum_nest.iteration_logic
  def _():
      Denom[0] += Output[a]

  
  accum_schedule =  accum_nest.create_schedule()
  a, = accum_schedule.get_indices()
  aa = accum_schedule.split(a, 8 * vector_size)
  aaa = accum_schedule.split(aa, 2 * vector_size)

  accum_plan = accum_schedule.create_action_plan()

  accum_plan.unroll(aa)
  accum_plan.vectorize(aaa)

  package.add_function(accum_plan, args=(Denom, Output), base_name="vectorized_2_accum")

def div():
  div_nest = acc.Nest(shape=(N,))
  j = div_nest.get_indices()


  @div_nest.iteration_logic
  def _():
      Output[j] /= Denom[0]


  div_schedule = div_nest.create_schedule()
  j, = div_schedule.get_indices()
  jj = div_schedule.split(j, 8 * vector_size)
  jjj = div_schedule.split(jj, 2 * vector_size)

  div_plan = div_schedule.create_action_plan()

  div_plan.unroll(jj)
  div_plan.vectorize(jjj)

  package.add_function(div_plan, args=(Denom, Output), base_name="vectorized_2_div")


max()
exp()
accum()
div()

package.build(
    name="vectorized_2",
    mode=acc.Package.Mode.DEBUG if DEV_MODE else acc.Package.Mode.RELEASE,
)
