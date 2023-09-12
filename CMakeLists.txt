cmake_minimum_required(VERSION 3.25)

get_property(gen_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT gen_multi AND NOT (CMAKE_BUILD_TYPE OR DEFINED ENV{CMAKE_BUILD_TYPE}))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release can have faster run time than Debug")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

project(hugr-mlir LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)
set(CMAKE_POSITION_INDEPENDENT_CODE On)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
include(AddMLIR)
include(AddLLVM)
include(TableGen)
include(CMakePrintHelpers)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()


add_library(mlir-interface INTERFACE)
target_include_directories(mlir-interface
  INTERFACE "$<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS};${LLVM_INCLUDE_DIRS}>"
)
