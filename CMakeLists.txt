cmake_minimum_required(VERSION 2.8.12)
project(Benchmark LANGUAGES CXX)

if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()
if(POLICY CMP0037)
  cmake_policy(SET CMP0037 NEW)
endif()
if(POLICY CMP0010)
  cmake_policy(SET CMP0010 NEW)
endif()
if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(
    CMAKE_BUILD_TYPE "Release"
    CACHE
      STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

option(USE_PYTORCH "Use pytorch C++ API" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(
    STATUS
      "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(
    DOWNLOAD
      "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
      "${CMAKE_BINARY_DIR}/conan.cmake"
    EXPECTED_HASH
      SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
    TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(REQUIRES
                      fmt/6.1.2
                      benchmark/1.6.0
                      openblas/0.3.17
                      nlohmann_json/3.9.1
                      clipp/1.2.3
                      xsimd/7.5.0
                      xtensor/0.23.10
                      GENERATORS
                      cmake_find_package)

conan_cmake_autodetect(settings)

conan_cmake_install(PATH_OR_REFERENCE
                    .
                    BUILD
                    missing
                    REMOTE
                    conan-center
                    SETTINGS
                    ${settings})

if(CMAKE_C_COMPILER MATCHES ".*(clang|emcc).*"
   OR CMAKE_C_COMPILER_ID MATCHES ".*(Clang|emcc).*")
  set(COMPILER_IS_CLANG TRUE)
endif()

if(CMAKE_COMPILER_IS_GNUCC
   OR CMAKE_COMPILER_IS_GNUCXX
   OR CMAKE_C_COMPILER MATCHES ".*(gcc|clang|emcc).*"
   OR CMAKE_C_COMPILER_ID MATCHES ".*(GCC|Clang|emcc).*")
  set(IS_GCC_LIKE TRUE)
else()
  set(IS_GCC_LIKE FALSE)
endif()

if(IS_GCC_LIKE AND NOT COMPILER_IS_CLANG)
  set(COMPILER_IS_GCC TRUE)
endif()

set(XTENSOR_USE_XSIMD ON) # see https://xtensor.readthedocs.io/en/latest/build-options.html#external-dependencies

if (USE_PYTORCH)
execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)
# message(STATUS "TorchPath = ${TorchPath}")

find_package(Torch HINTS ${TorchPath})
endif (USE_PYTORCH)

find_package(fmt REQUIRED)
find_package(benchmark REQUIRED)
find_package(clipp REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(xsimd REQUIRED)
find_package(xtensor REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Development)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

 



if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(MSVC)
  # Perform extremely aggressive optimization on Release builds: Flags on Visual
  # Studio 2010 and newer: Runtime library: Multi-threaded (/MT) as opposed to
  # default 'Multi-threaded DLL' - static runtime library allows better LTCG
  # inlining opportunities. Optimization: Full Optimization (/Ox) Inline
  # Function Expansion: Any Suitable (/Ob2) Enable Intrinsic Functions: Yes
  # (/Oi) Favor Size Or Speed: Favor fast code (/Ot) Enable Fiber-Safe
  # Optimizations: Yes (/GT) Enable String Pooling: Yes (/GF) Buffer Security
  # Check: No (/GS-) Floating Point Model: Fast (/fp:fast) Enable Floating Point
  # Exceptions: No (/fp:except-) Build with Multiple Processes (/MP)
  add_definitions("/Ox /Ob2 /Oi /Ot /GT /GF /GS- /fp:fast /fp:except- /MP")
  add_definitions(/arch:AVX)
elseif(IS_GCC_LIKE)
  # http://gcc.gnu.org/onlinedocs/gcc-4.8.2/gcc/i386-and-x86-64-Options.html#i38
  # 6-and-x86-64-Options add_definitions(-march=corei7-avx -mtune=corei7-avx)
  add_definitions(-march=native)
  add_definitions(-ffp-contract=fast)
  add_definitions(-fno-math-errno)
  add_definitions(-fno-semantic-interposition)
  add_definitions(-flto) 
  add_definitions(-fno-omit-frame-pointer)
endif()

if(COMPILER_IS_CLANG)
  add_definitions(-DCOMPILER_IS_CLANG=1)
  add_definitions(-Rpass=loop-vectorize)
  add_definitions(-ferror-limit=1)
endif(COMPILER_IS_CLANG)


macro(add_robocode_ MODULE_NAME IMPLEMENTATION_NAME)
  set(_PythonGeneratorFile ${IMPLEMENTATION_NAME}.py)
  set(_OutputObjectFile ${IMPLEMENTATION_NAME}.o)
  set(_HatFile ${IMPLEMENTATION_NAME}.hat)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_OutputObjectFile}"
    PRE_BUILD
    COMMAND "${PYTHON_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${_PythonGeneratorFile}"
            --argument
    COMMAND ${CMAKE_OBJDUMP}
            --disassemble
            # --demangle
            --visualize-jumps
            --no-show-raw-insn
            # -l
            # --wide
            # -S 
            "${CMAKE_CURRENT_BINARY_DIR}/${_OutputObjectFile}" >
                   ${CMAKE_CURRENT_BINARY_DIR}/${IMPLEMENTATION_NAME}.s
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_PythonGeneratorFile}")
  add_custom_target(robocode_${MODULE_NAME}_${IMPLEMENTATION_NAME} ALL
                    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_OutputObjectFile}"
                            "${CMAKE_CURRENT_BINARY_DIR}/${_HatFile}")
  add_dependencies(${MODULE_NAME} "robocode_${MODULE_NAME}_${IMPLEMENTATION_NAME}")
  target_link_libraries(${MODULE_NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/${_OutputObjectFile}")
  target_sources(${MODULE_NAME} PRIVATE robocode_${IMPLEMENTATION_NAME}.cpp)
endmacro()

macro(add_robocode MODULE_NAME)
  set(robocode_implementations ${ARGN}) 
  foreach(robocode_implementation IN LISTS robocode_implementations)
    add_robocode_(${MODULE_NAME} ${robocode_implementation})
  endforeach()
endmacro()


include_directories(SYSTEM ${CONAN_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src/utils)
link_libraries(
  ${CONAN_LIBS}
  benchmark::benchmark
  nlohmann_json::nlohmann_json
  fmt::fmt
  clipp::clipp
  OpenBLAS::OpenBLAS
  xsimd::xsimd
  xtensor::xtensor 
)

# add_subdirectory(src/softmax)
# add_subdirectory(src/batched_softmax)
add_subdirectory(src/scaled_dot_product_attention)