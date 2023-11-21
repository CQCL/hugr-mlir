add_mlir_tool(hugr-mlir-opt
  hugr-mlir-opt.cpp
)
target_link_libraries(hugr-mlir-opt
  PRIVATE
  HugrMLIRDialect
  HugrMLIRAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIROptLib
)
set_target_properties(hugr-mlir-opt
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${HUGR_MLIR_TOOLS_DIR}"
)
mlir_check_all_link_libraries(hugr-mlir-opt)
install(TARGETS hugr-mlir-opt
  RUNTIME
)
