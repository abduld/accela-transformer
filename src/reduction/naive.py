#!/usr/bin/env python3

### [import-package]
import robopy as acc
### [import-package]

### [declare-input-length]
N = 2 ** 20 
### [declare-input-length]


### [declare-inputs]
Input = acc.Array(
    role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(N,)
)
Sum = acc.Array(
    role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(1,)
) 
### [declare-inputs]
 
 
### [declare-nest]
accum_nest = acc.Nest(shape=(N,))
### [declare-nest]

### [get-nest-indices]
i = accum_nest.get_indices()
### [get-nest-indices]

### [declare-iteration-logic]
@accum_nest.iteration_logic
def _():
    Sum[0] += Input[i]
### [declare-iteration-logic]

### [create-package]
package = acc.Package()
package.add_function(accum_nest, args=(Sum, Input), base_name="naive")
### [create-package]

### [build-package]
package.build(
    name="naive",
    format=acc.Package.Format.HAT,
    mode=acc.Package.Mode.RELEASE,
)
### [build-package]
