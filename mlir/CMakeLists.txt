
# A target to add all include directories for this project

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
