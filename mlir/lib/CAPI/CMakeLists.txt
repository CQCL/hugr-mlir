add_mlir_public_c_api_library(HugrMLIRDialectCAPI
  Dialects.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  HugrMLIRDialect
  MLIRCAPIIR
)

add_mlir_public_c_api_library(HugrMLIRTranslateCAPI
  Translate.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  HugrMLIRTranslate
)

add_mlir_public_c_api_library(HugrMLIRAnalysisCAPI
  Analysis.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  HugrMLIRAnalysis
)

add_mlir_public_c_api_library(HugrMLIRTransformsCAPI
  Transforms.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  HugrMLIRTransforms
)

add_mlir_public_c_api_library(HugrMLIRSupportCAPI
  Support.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  hugr-mlir-headers-interface
)


get_property(_capi_libraries GLOBAL PROPERTY MLIR_ALL_LIBS)
list(FILTER _capi_libraries INCLUDE REGEX "^MLIRCAPI")

message(STATUS "capi libraries: ${_capi_libraries}")
add_mlir_aggregate(HugrMLIR-C
  SHARED
  EMBED_LIBS
  ${_capi_libraries}
  HugrMLIRDialectCAPI
  HugrMLIRTranslateCAPI
  HugrMLIRAnalysisCAPI
  HugrMLIRTransformsCAPI
  HugrMLIRSupportCAPI
)
set_property(TARGET HugrMLIR-C APPEND PROPERTY INSTALL_RPATH "${LLVM_LIBRARY_DIR}")
set_property(TARGET HugrMLIR-C PROPERTY OUTPUT_NAME "HugrMLIRC")


if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(HugrMLIR-C PRIVATE "-Wl,-exclude-libs,ALL")
else()
  if(NOT CMAKE_C_VISIBILITY_PRESET STREQUAL "hidden" OR NOT CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden")
    message(STATUS "HugrMLIR-C on this platform exports all symbols. Recommend building with CMAKE_(C|CXX)_VISIBILITY_PRESET=hidden or implement filtering support.")
  endif()
endif()

cmake_print_properties(TARGETS HugrMLIR-C PROPERTIES
  BUILD_WITH_INSTALL_RPATH
  BUILD_RPATH
  INSTALL_RPATH
)
