set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
)

set(LIBS
  HugrMLIRDialect

  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect

  MLIRDialect
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRIR
)

add_executable(hugr-mlir-lsp-server
  hugr-mlir-lsp-server.cpp
)
set_target_properties(hugr-mlir-lsp-server
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${HUGR_MLIR_TOOLS_DIR}"
)
target_link_libraries(hugr-mlir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(hugr-mlir-lsp-server)
install(TARGETS hugr-mlir-lsp-server
  RUNTIME
)

