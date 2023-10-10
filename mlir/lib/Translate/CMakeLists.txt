get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
add_mlir_library(HugrMLIRTranslate
    HugrTranslate.cpp
    LINK_LIBS PUBLIC
    ${dialect_libs}
    ${translation_libs}
    MLIRTranslateLib
)
target_link_libraries(HugrMLIRTranslate
  PUBLIC hugr-mlir-headers-interface
)

mlir_check_all_link_libraries(hugr-mlir-translate)
