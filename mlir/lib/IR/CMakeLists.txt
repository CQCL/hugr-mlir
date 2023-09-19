add_mlir_dialect_library(HugrMLIRDialect
  HugrDialect.cpp
  HugrOps.cpp
  HugrAttrs.cpp
  HugrTypes.cpp
  HugrEnums.cpp
  HugrTypeInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ../../include/hugr-mlir

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRControlFlowDialect
  MLIRFunctionInterfaces
)
target_link_libraries(HugrMLIRDialect
  PUBLIC hugr-mlir-headers-interface
)

