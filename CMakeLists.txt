cmake_minimum_required(VERSION 2.8.12)
project(Benchmark LANGUAGES CXX )


if (POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()
if (POLICY CMP0037)
  cmake_policy(SET CMP0037 NEW)
endif()
if (POLICY CMP0010)
  cmake_policy(SET CMP0010 NEW)
endif()
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif()


SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()


option(USE_TIMERS "Instrument the code by using kernel timers" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake"
            EXPECTED_HASH SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
            TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(REQUIRES
                    fmt/6.1.2
                    benchmark/1.5.6
                    openblas/0.3.17
                    nlohmann_json/3.9.1
                    clipp/1.2.3
                  GENERATORS cmake_find_package)

conan_cmake_autodetect(settings)

conan_cmake_install(PATH_OR_REFERENCE .
                BUILD missing
                REMOTE conan-center
                SETTINGS ${settings})

find_package(fmt REQUIRED)
find_package(benchmark REQUIRED)
find_package(clipp REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP)
find_package(PythonInterp REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


add_subdirectory(softmax)
