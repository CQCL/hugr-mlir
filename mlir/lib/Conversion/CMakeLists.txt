add_mlir_library(HugrMLIRConversion
  TypeConverter.cpp
  PreConvertHugrFuncPass.cpp
  ConvertHugrPass.cpp
  ConvertHugrFuncPass.cpp
  ConvertHugrExtArithPass.cpp

  DEPENDS
  HugrMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  HugrMLIRDialect
  HugrMLIRTransforms
  MLIRRewrite
  MLIRTransforms
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRSCFTransforms
  MLIRControlFlowDialect
  # MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRMathDialect
)
