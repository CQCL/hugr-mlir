cmake_minimum_required(VERSION 3.25)

# default to Release
get_property(gen_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT gen_multi AND NOT (CMAKE_BUILD_TYPE OR DEFINED ENV{CMAKE_BUILD_TYPE}))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(HUGR_MLIR_INCLUDE_DOCS "Build docs as part of ALL target")

project(hugr-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS On)
# We will generate shared libraries by linking to static libraries, so ensure
# said static libraries are built with -fPIC
# TODO if we add an add_hugr_mlir_library function, have that set the
# POSITION_INDEPENDENT_CODE property instead
set(CMAKE_POSITION_INDEPENDENT_CODE On)

include(CMakePrintHelpers)
include(CTest)

add_custom_target(check)
set(_doc_all "")
if(HUGR_MLIR_INCLUDE_DOCS)
  set(_doc_all "ALL")
endif()
add_custom_target(doc ${_doc_all})

add_subdirectory(docs)
add_subdirectory(mlir)

block()
  # This is based on https://michael-f-bryan.github.io/rust-ffi-guide/setting_up.html
  set(HUGR_MLIR_CARGO_BUILD_FLAGS "${HUGR_MLIR_CARGO_BUILD_FLAGS}")
  set(HUGR_MLIR_CARGO_TEST_FLAGS "${HUGR_MLIR_CARGO_TEST_FLAGS}")
  # set(HUGR_MLIR_CARGO_DOC_FLAGS "${HUGR_MLIR_CARGO_DOC_FLAGS}")

  set(HUGR_MLIR_CONFIG_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target")
  set(HUGR_MLIR_CONFIG_PROFILE_RELEASE_DEBUG "false")
  set(HUGR_MLIR_CONFIG_RUSTFLAGS)
  set(_cargo_config_dir "${CMAKE_CURRENT_SOURCE_DIR}/.cargo")
  set(_cargo_config "${_cargo_config_dir}/config.toml")
  set(_cargo_lock Cargo.lock)

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(_outdir "${HUGR_MLIR_TARGET_DIR}/debug")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
      list(APPEND HUGR_MLIR_CARGO_BUILD_FLAGS "--release")
      list(APPEND HUGR_MLIR_CARGO_TEST_FLAGS "--release")
      set(_outdir "${HUGR_MLIR_TARGET_DIR}/release")
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      list(APPEND HUGR_MLIR_CARGO_FLAGS "--release")
      set(HUGR_MLIR_CONFIG_PROFILE_RELEASE_DEBUG "true")
      set(_outdir "${HUGR_MLIR_TARGET_DIR}/release")
  endif ()

  configure_file(
      "${_cargo_config}.in"
      "${CMAKE_CURRENT_BINARY_DIR}/config.toml.in"
      @ONLY)

  file(GENERATE OUTPUT "${_cargo_config}"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/config.toml.in"
  )

  add_custom_command(
      COMMENT "Compiling hugr-mlir crate"
      OUTPUT
      "${HUGR_MLIR_TARGET_DIR}/libhugr_mlir.rlib"
      DEPENDS "${_cargo_config}" HugrMLIR-C
      COMMAND cargo build ${HUGR_MLIR_CARGO_BUILD_FLAGS}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      DEPFILE
      "${HUGR_MLIR_TARGET_DIR}/libhugr_mlir.d"
      BYPRODUCTS
      "${HUGR_MLIR_TARGET_DIR}/libhugr_mlir.d"
      "${_cargo_lock}"
      USES_TERMINAL
  )
  add_custom_target(hugr_mlir_crate_target ALL
      DEPENDS "${HUGR_MLIR_TARGET_DIR}/libhugr_mlir.rlib"
  )
  add_custom_command(COMMENT "Building hugr-mlir crate docs"
      OUTPUT
      "${CMAKE_CURRENT_BINARY_DIR}/target/doc/settings.html"
      COMMAND cargo doc
      DEPENDS hugr_mlir_crate_target
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      USES_TERMINAL
  )
  add_custom_target(hugr_mlir_crate_doc
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/target/doc/settings.html"
  )
  add_dependencies(doc hugr_mlir_crate_doc)

  add_custom_target(hugr_mlir_crate_test
    COMMENT "Testing hugr-mlir crate"
    DEPENDS "${_cargo_config}" HugrMLIR-C
    COMMAND cargo test ${HUGR_MLIR_CARGO_TEST_FLAGS}
    USES_TERMINAL
  )

  add_dependencies(check hugr_mlir_crate_test)
endblock()

block()
# set(_venv "${CMAKE_CURRENT_BINARY_DIR}/venv")
# add_custom_command(COMMENT "Creating python virtual env"
#   COMMAND "${PYTHON3_Executable}" -m venv "${_venv}"
#   OUTPUT "${_venv}/pyvenv.cfg"
# )
# set(_venv_python "${_venv}/bin/python")
# set(_venv_pip "${_venv}/bin/pip")
# add_custom_target(python-venv DEPENDS "${_venv}/pyvenv.cfg")

# add_custom_target(build-guppy
#   COMMENT "Installing guppy into venv"
#   COMMAND "${_venv_pip}" install -r guppy/requirements.txt
#   COMMAND "${_venv_pip}" install -e 'guppy[dev]'
#   COMMAND cd guppy/validator && . "${_venv}/bin/activate" && maturin develop
#   DEPENDS python-venv "${CMAKE_CURRENT_SOURCE_DIR}/guppy/requirements.txt"
#   USES_TERMINAL
#   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
# )

# set(_guppy_export_dir "${CMAKE_CURRENT_BINARY_DIR}/guppy-exports")
# add_custom_target(test-guppy
#   COMMENT "Running guppy tests"
#   COMMAND ${CMAKE_COMMAND} -E make_directory "${_guppy_export_dir}"
#   COMMAND rm -rf "${_guppy_export_dir}/*"
#   COMMAND ${_venv}/bin/pytest "${CMAKE_CURRENT_SOURCE_DIR}/guppy/tests" --export-test-cases "${_guppy_export_dir}"
#   DEPENDS build-guppy
#   USES_TERMINAL
# )

# add_dependencies(check test-guppy)
endblock()
