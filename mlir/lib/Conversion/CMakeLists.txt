add_mlir_library(HugrMLIRConversion
  TypeConverter.cpp
  ConvertHugrPass.cpp
  ConvertHugrFuncPass.cpp

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
)
