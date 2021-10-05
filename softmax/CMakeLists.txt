set(MODULE_NAME softmax)

add_executable(${MODULE_NAME}
  main.cpp
)
add_robocode(${MODULE_NAME} softmax_naive)
target_link_libraries(${MODULE_NAME} PRIVATE
        ${CONAN_LIBS}
        benchmark::benchmark
        nlohmann_json::nlohmann_json
        fmt::fmt
        clipp::clipp
        OpenBLAS::OpenBLAS
        ${CONAN_LIBS})
target_compile_definitions(${MODULE_NAME} -DMODULE_NAME=${MODULE_NAME})
target_include_directories(${MODULE_NAME} SYSTEM PRIVATE ${CONAN_INCLUDE_DIRS})

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION bin)
