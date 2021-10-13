set(MODULE_NAME scaled_dot_product_attention)

add_executable(${MODULE_NAME}
  main.cpp
  cpp_naive.cpp
)
add_robocode(${MODULE_NAME} naive)

target_compile_definitions(${MODULE_NAME} PRIVATE -DMODULE_NAME=${MODULE_NAME})
target_include_directories(${MODULE_NAME} SYSTEM PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)


set_target_properties(${MODULE_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
