add_executable(hugr-mlir-translate
  hugr-mlir-translate.cpp
)
target_link_libraries(hugr-mlir-translate
  PRIVATE
  MLIRTranslateLib
  HugrMLIRDialect
  # hugr-rs-bridge
)
set_target_properties(hugr-mlir-translate
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${HUGR_MLIR_TOOLS_DIR}"
)
llvm_update_compile_flags(hugr-mlir-translate)
mlir_check_all_link_libraries(hugr-mlir-translate)

install(TARGETS hugr-mlir-translate
  RUNTIME
)
