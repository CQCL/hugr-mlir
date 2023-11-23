include(AddMLIR)
include(AddLLVM)
include(TableGen)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

if(HUGR_MLIR_SANITIZER)
  add_compile_options("-fsanitize=${HUGR_MLIR_SANITIZER}")
  add_link_options("-fsanitize=${HUGR_MLIR_SANITIZER}")
endif()

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
add_subdirectory(include/hugr-mlir/Analysis)
add_subdirectory(lib/IR)
add_subdirectory(lib/CAPI)
add_subdirectory(lib/Translate)
add_subdirectory(lib/Analysis)
add_subdirectory(tools/hugr-mlir-opt)
add_subdirectory(tools/hugr-mlir-translate)
add_subdirectory(tools/hugr-mlir-lsp-server)
add_subdirectory(python)
add_subdirectory(test)
