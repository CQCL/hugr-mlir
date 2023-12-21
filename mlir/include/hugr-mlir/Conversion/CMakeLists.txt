set(MLIR_BINARY_DIR "${PROJECT_BINARY_DIR}")

# Include directories for .td files setting the directory property
# (i.e. include_directories) is the only way to pass these
get_target_property(_extra_includes hugr-mlir-headers-interface INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_extra_includes_mlir mlir-headers-interface INTERFACE_INCLUDE_DIRECTORIES)
list(APPEND _extra_includes ${_extra_includes_mlir})

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HugrConversion EXTRA_INCLUDES ${_extra_includes})
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix HugrConversion EXTRA_INCLUDES ${_extra_includes})
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix HugrConversion EXTRA_INCLUDES ${_extra_includes})
add_public_tablegen_target(HugrMLIRConversionPassIncGen)

add_mlir_doc(Passes HugrConversionPasses ./ -gen-pass-doc EXTRA_INCLUDES ${_extra_includes})
