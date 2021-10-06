set(MODULE_NAME softmax)

add_executable(${MODULE_NAME}
  main.cpp
  cpp_naive.cpp
  cpp_simd_xsimd.cpp
  cpp_xtensor.cpp
  cpp_simd_openmp.cpp
)
add_robocode(${MODULE_NAME} naive vectorized)

target_compile_definitions(${MODULE_NAME} PRIVATE -DMODULE_NAME=${MODULE_NAME})
target_include_directories(${MODULE_NAME} SYSTEM PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)

set_target_properties(${MODULE_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

