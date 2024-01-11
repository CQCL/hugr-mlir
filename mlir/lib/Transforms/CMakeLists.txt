add_mlir_library(HugrMLIRTransforms
  Utils.cpp

  DEPENDS
  HugrMLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  HugrMLIRDialect
  MLIRRewrite
  MLIRTransforms
  MLIRIndexDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
)
