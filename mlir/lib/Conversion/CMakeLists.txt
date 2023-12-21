add_mlir_library(HugrMLIRConversion
  TypeConverter.cpp
  ConvertHugrPass.cpp

  DEPENDS
  HugrMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  HugrMLIRDialect
  MLIRRewrite
  MLIRTransforms
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRSCFTransforms
  MLIRControlFlowDialect
)
