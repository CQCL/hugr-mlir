add_mlir_library(HugrMLIRAnalysis
  FreeAllocAnalysis.cpp
  Passes.cpp

  DEPENDS
  HugrMLIRAnalyisisPassIncGen

  LINK_LIBS PUBLIC
  HugrMLIRDialect
)
target_link_libraries(HugrMLIRDialect
  PUBLIC hugr-mlir-headers-interface
)
