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


add_mlir_aggregate(HugrMLIR-C
  SHARED
  EMBED_LIBS
  HugrMLIRDialectCAPI
  HugrMLIRTranslateCAPI
)
set_property(TARGET HugrMLIR-C APPEND PROPERTY INSTALL_RPATH "${LLVM_LIBRARY_DIR}")

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
