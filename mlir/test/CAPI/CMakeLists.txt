add_executable(test-hugr-mlir-capi
  TestHugrMlirCAPI.c
)

target_link_libraries(test-hugr-mlir-capi
  PRIVATE
  HugrMLIRDialectCAPI
  MLIRCAPIIR
  MLIRCAPIDebug
)
set_target_properties(test-hugr-mlir-capi
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)
llvm_update_compile_flags(test-hugr-mlir-capi)
mlir_check_all_link_libraries(test-hugr-mlir-capi)

# TODO We don't want to install tests in general. We do it here so that we can
# verify that tests work when run from the install tree
install(TARGETS test-hugr-mlir-capi
  TYPE RUNTIME
)

