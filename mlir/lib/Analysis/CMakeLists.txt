add_mlir_library(HugrMLIRAnalysis
  FreeAllocAnalysis.cpp
  LINK_LIBS PUBLIC
  HugrMLIRDialect
)
target_link_libraries(HugrMLIRDialect
  PUBLIC hugr-mlir-headers-interface
)
