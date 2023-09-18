# Include directories for .td files setting the directory property
# (i.e. include_directories) is the only way to pass these
cmake_print_properties(TARGETS hugr-mlir-headers-interface PROPERTIES INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_extra_includes hugr-mlir-headers-interface INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_extra_includes_mlir mlir-headers-interface INTERFACE_INCLUDE_DIRECTORIES)
list(APPEND _extra_includes ${_extra_includes_mlir})
message(STATUS "tablegen extra includes: ${_extra_includes}")

# add_mlir_doc puts docs in ${MLIR_BINARY_DIR}/docs
# add_mlir_doc puts docs in ${MLIR_BINARY_DIR}/docs
set(MLIR_BINARY_DIR "${PROJECT_BINARY_DIR}")

# add_mlir_doc puts docs in ${MLIR_BINARY_DIR}/docs
set(MLIR_BINARY_DIR "${PROJECT_BINARY_DIR}")

set(LLVM_TARGET_DEFINITIONS HugrDialect.td)
mlir_tablegen(HugrDialect.h.inc --gen-dialect-decls -dialect=hugr EXTRA_INCLUDES ${_extra_includes})
mlir_tablegen(HugrDialect.cpp.inc --gen-dialect-defs -dialect=hugr EXTRA_INCLUDES ${_extra_includes})

set(LLVM_TARGET_DEFINITIONS HugrOps.td)
mlir_tablegen(HugrOps.h.inc --gen-op-decls EXTRA_INCLUDES ${_extra_includes})
mlir_tablegen(HugrOps.cpp.inc --gen-op-defs EXTRA_INCLUDES ${_extra_includes})

set(LLVM_TARGET_DEFINITIONS HugrAttrs.td)
mlir_tablegen(HugrAttrs.h.inc --gen-attrdef-decls --attrdefs-dialect=hugr EXTRA_INCLUDES ${_extra_includes})
mlir_tablegen(HugrAttrs.cpp.inc --gen-attrdef-defs --attrdefs-dialect=hugr EXTRA_INCLUDES ${_extra_includes})

set(LLVM_TARGET_DEFINITIONS HugrTypes.td)
mlir_tablegen(HugrTypes.h.inc --gen-typedef-decls --typedefs-dialect=hugr EXTRA_INCLUDES ${_extra_includes})
mlir_tablegen(HugrTypes.cpp.inc --gen-typedef-defs --typedefs-dialect=hugr EXTRA_INCLUDES ${_extra_includes})

set(LLVM_TARGET_DEFINITIONS HugrEnums.td)
mlir_tablegen(HugrEnums.h.inc --gen-enum-decls EXTRA_INCLUDES ${_extra_includes})
mlir_tablegen(HugrEnums.cpp.inc --gen-enum-defs EXTRA_INCLUDES ${_extra_includes})

set(LLVM_TARGET_DEFINITIONS HugrTypeInterfaces.td)
mlir_tablegen(HugrTypeInterfaces.h.inc -gen-type-interface-decls EXTRA_INCLUDES ${_extra_includes})
mlir_tablegen(HugrTypeInterfaces.cpp.inc -gen-type-interface-defs EXTRA_INCLUDES ${_extra_includes})
# TODO If we need any AttrInterfaces or OpInterfaces add these with the same pattern

add_public_tablegen_target(HugrMLIRIncGen)
add_dependencies(hugr-mlir-headers-interface HugrMLIRIncGen)

add_mlir_doc(HugrOps HugrDialect Hugr/ -gen-dialect-doc -dialect=hugr EXTRA_INCLUDES ${_extra_includes})
# The interfaces are not part of the dialect, and aren't included in -gen-dialect-doc
add_mlir_doc(HugrTypeInterfaces HugrTypeInterfaces Hugr/ --gen-type-interface-docs  -dialect=hugr EXTRA_INCLUDES ${_extra_includes})

add_dependencies(doc mlir-doc)

install(FILES ${MLIR_BINARY_DIR}/docs/Hugr/HugrDialect.md
  TYPE DOC
  RENAME Hugr/HugrDialect.md
)
