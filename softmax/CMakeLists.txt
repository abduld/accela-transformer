set(MODULE_NAME softmax)

macro(add_robocode MODULE_NAME IMPLEMENTATION_NAME)
  set(_PythonGeneratorFile ${IMPLEMENTATION_NAME}.py)
  set(_OutputObjectFile ${IMPLEMENTATION_NAME}.o)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_OutputObjectFile}" PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${_PythonGeneratorFile}" "${CMAKE_CURRENT_BINARY_DIR}/${_PythonGeneratorFile}"
      COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/${_PythonGeneratorFile}" --argument
      COMMAND ${CMAKE_OBJDUMP} --disassemble --demangle --wide -S "${CMAKE_CURRENT_BINARY_DIR}/${_OutputObjectFile}" > ${CMAKE_CURRENT_BINARY_DIR}/${IMPLEMENTATION_NAME}.s
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      DEPENDS ${_PythonGeneratorFile})
  add_custom_target(robocode_${IMPLEMENTATION_NAME} ALL
      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_OutputObjectFile}")
  target_link_libraries(${MODULE_NAME} PRIVATE robocode_${IMPLEMENTATION_NAME})
endmacro()


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
        OpenBLAS::OpenBLAS)

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION bin)
