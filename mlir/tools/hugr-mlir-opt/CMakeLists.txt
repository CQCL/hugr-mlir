add_executable(hugr-mlir-opt
  hugr-mlir-opt.cpp
)
target_link_libraries(hugr-mlir-opt
  PRIVATE
  HugrMLIRDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIROptLib
)
set_target_properties(hugr-mlir-opt
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${HUGR_MLIR_TOOLS_DIR}"
)
llvm_update_compile_flags(hugr-mlir-opt)
mlir_check_all_link_libraries(hugr-mlir-opt)
install(TARGETS hugr-mlir-opt
  RUNTIME
)
