cmake_minimum_required(VERSION 3.25)

# default to Release
get_property(gen_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT gen_multi AND NOT (CMAKE_BUILD_TYPE OR DEFINED ENV{CMAKE_BUILD_TYPE}))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release can have faster run time than Debug")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

project(hugr-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS On)
# We will generate shared libraries by linking to static libraries, so ensure
# said static libraries are built with -fPIC
# TODO if we add an add_hugr_mlir_library function, have that set the
# POSITION_INDEPENDENT_CODE property instead
set(CMAKE_POSITION_INDEPENDENT_CODE On)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})

set(LLVM_EXTERNAL_LIT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/lit-wrapped.py")
set(LLVM_LIT_ARGS "-v" CACHE STRING "")

include(AddMLIR)
include(AddLLVM)
include(TableGen)
include(CMakePrintHelpers)
include(CTest)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

add_custom_target(check)

# llvm doesn't provide targets with interface include directories, so we'll use
# this
add_library(mlir-headers-interface INTERFACE)
# lists inside BUILD_INTERFACE generator expressions cause trouble
foreach(_x ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_include_directories(mlir-headers-interface
  INTERFACE "$<BUILD_INTERFACE:${_x}>"
)
endforeach()

message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIR: ${MLIR_INCLUDE_DIR}")
message(STATUS "LLVM_INCLUDE_DIR: ${LLVM_INCLUDE_DIR}")
target_compile_features(mlir-headers-interface INTERFACE cxx_std_17)
cmake_print_properties(TARGETS mlir-headers-interface PROPERTIES INTERFACE_INCLUDE_DIRECTORIES)

add_subdirectory(docs)
add_subdirectory(mlir)
