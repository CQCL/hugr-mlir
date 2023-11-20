cmake_minimum_required(VERSION 3.25)

# default to Release
get_property(gen_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT gen_multi AND NOT (CMAKE_BUILD_TYPE OR DEFINED ENV{CMAKE_BUILD_TYPE}))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(HUGR_MLIR_INCLUDE_DOCS "Build docs as part of ALL target")

project(hugr-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS On)
# We will generate shared libraries by linking to static libraries, so ensure
# said static libraries are built with -fPIC
# TODO if we add an add_hugr_mlir_library function, have that set the
# POSITION_INDEPENDENT_CODE property instead
set(CMAKE_POSITION_INDEPENDENT_CODE On)

include(CMakePrintHelpers)
include(CTest)

add_custom_target(check)
set(_doc_all "")
if(HUGR_MLIR_INCLUDE_DOCS)
  set(_doc_all "ALL")
endif()
add_custom_target(doc ${_doc_all})

add_subdirectory(docs)
add_subdirectory(mlir)
add_subdirectory(hugr-mlir)
