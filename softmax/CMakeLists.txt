set(MODULE_NAME softmax)

add_executable(${MODULE_NAME}
  main.cpp
  cpp_naive.cpp
  cpp_simd_xsimd.cpp
  robocode_naive.cpp
)
add_robocode(${MODULE_NAME} softmax_naive)
target_link_libraries(${MODULE_NAME} PRIVATE
        ${CONAN_LIBS}
        benchmark::benchmark
        nlohmann_json::nlohmann_json
        fmt::fmt
        clipp::clipp
        OpenBLAS::OpenBLAS
        xsimd::xsimd)
# target_compile_definitions(${MODULE_NAME} -DMODULE_NAME=${MODULE_NAME})
target_include_directories(${MODULE_NAME} SYSTEM PRIVATE
  ${CONAN_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION bin)
