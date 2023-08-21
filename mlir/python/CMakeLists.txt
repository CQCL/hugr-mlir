include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=hugr_mlir.")

################################################################################
# Sources
################################################################################


# There does not seem to be a better way to add include directories to these targets
get_target_property(_extra_includes hugr-mlir-headers-interface INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_extra_includes_mlir mlir-headers-interface INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${_extra_includes} ${_extra_includes_mlir})

declare_mlir_python_sources(HugrMLIRPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
)

declare_mlir_python_extension(HugrMLIRPythonSources.Extension
  MODULE_NAME _hugr
  ADD_TO_PARENT HugrMLIRPythonSources
  SOURCES
    HugrExtension.cpp
  PRIVATE_LINK_LIBS
    hugr-mlir-headers-interface
  EMBED_CAPI_LINK_LIBS
    HugrMLIRDialectCAPI
)


declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HugrMLIRPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/Hugr.td
  SOURCES
    dialects/hugr.py
  DIALECT_NAME hugr)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(HugrMLIRPythonCAPI
  INSTALL_COMPONENT HugrMLIRPythonModules
  INSTALL_DESTINATION python_packages/hugr_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${HUGR_MLIR_PYTHON_PACKAGES_DIR}/hugr_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    HugrMLIRPythonSources
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(HugrMLIRPythonModules
  ROOT_PREFIX "${HUGR_MLIR_PYTHON_PACKAGES_DIR}/hugr_mlir"
  INSTALL_PREFIX "python_packages/hugr_mlir"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    HugrMLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    HugrMLIRPythonCAPI
)


