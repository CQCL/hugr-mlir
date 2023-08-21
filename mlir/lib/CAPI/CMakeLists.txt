add_mlir_public_c_api_library(HugrMLIRDialectCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  HugrMLIRDialect
  MLIRCAPIIR
)
