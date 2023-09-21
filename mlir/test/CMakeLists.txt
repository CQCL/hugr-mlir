llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

# If we're in an install tree, otherwise...
# We use this to find the mlir python packages
if(MLIR_INSTALL_PREFIX)
  set(MLIR_OBJ_ROOT "${MLIR_INSTALL_PREFIX}")
else()
  set(MLIR_OBJ_ROOT "${LLVM_BINARY_DIR}/tools/mlir")
endif()

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(HUGR_MLIR_TEST_DEPENDS
        FileCheck count not
        hugr-mlir-opt
        test-hugr-mlir-capi
        )

if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND HUGR_MLIR_TEST_DEPENDS HugrMLIRPythonModules)
endif()

add_lit_testsuite(check-hugr-mlir "Running the hugr-mlir regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${HUGR_MLIR_TEST_DEPENDS}
)

set_target_properties(check-hugr-mlir PROPERTIES FOLDER "Tests")
add_dependencies(check check-hugr-mlir)

add_lit_testsuites(HUGR-MLIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${HUGR_MLIR_TEST_DEPENDS})

# llvm lit suites don't have an integration with CTest, this makes it work
add_test(NAME hugr-mlir-lit
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-hugr-mlir
)

add_subdirectory(CAPI)
