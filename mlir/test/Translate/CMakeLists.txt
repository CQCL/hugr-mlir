add_executable(test-import-example-hugr
  test-import-example-hugr.cpp
)

target_link_libraries(test-import-example-hugr
  PRIVATE
  hugr-rs-bridge
  HugrMLIRDialect
)
set_target_properties(test-import-example-hugr
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)
llvm_update_compile_flags(test-import-example-hugr)
mlir_check_all_link_libraries(test-import-example-hugr)

# TODO We don't want to install tests in general. We do it here so that we can
# verify that tests work when run from the install tree
install(TARGETS test-import-example-hugr
  TYPE RUNTIME
)

