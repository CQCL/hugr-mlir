add_mlir_library(HugrMLIRTransforms
  LowerHugrPass.cpp

  DEPENDS
  HugrMLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  HugrMLIRDialect
)
