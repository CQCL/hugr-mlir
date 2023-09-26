cmake_minimum_required(VERSION 3.25)

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

set(LLVM_EXTERNAL_LIT "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/lit-wrapped.py")
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
# We add docs to the ALL target for convenience. This could be controlled by an
# option
add_custom_target(doc ALL)

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

add_library(hugr-mlir-headers-interface INTERFACE)
target_link_libraries(hugr-mlir-headers-interface
  INTERFACE mlir-headers-interface
)
target_include_directories(hugr-mlir-headers-interface
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

set(HUGR_MLIR_TOOLS_DIR "${PROJECT_BINARY_DIR}/bin")
set(HUGR_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")

add_subdirectory(include/hugr-mlir/IR)
add_subdirectory(lib/IR)
add_subdirectory(lib/CAPI)
add_subdirectory(tools/hugr-mlir-opt)
add_subdirectory(tools/hugr-mlir-translate)
add_subdirectory(tools/hugr-mlir-lsp-server)
add_subdirectory(python)
add_subdirectory(test)
