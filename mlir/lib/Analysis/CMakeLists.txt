add_mlir_library(HugrMLIRAnalysis
  FreeAllocAnalysis.cpp
  VerifyLinearityPass.cpp

  DEPENDS
  HugrMLIRAnalyisisPassIncGen

  LINK_LIBS PUBLIC
  HugrMLIRDialect
)
