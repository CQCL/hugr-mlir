# This is based on https://michael-f-bryan.github.io/rust-ffi-guide/setting_up.html
set(HUGR_MLIR_CARGO_BUILD_FLAGS "${HUGR_MLIR_CARGO_BUILD_FLAGS}")
set(HUGR_MLIR_CARGO_TEST_FLAGS "${HUGR_MLIR_CARGO_TEST_FLAGS}")
# set(HUGR_MLIR_CARGO_DOC_FLAGS "${HUGR_MLIR_CARGO_DOC_FLAGS}")

set(HUGR_MLIR_CONFIG_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target")
set(HUGR_MLIR_CONFIG_PROFILE_RELEASE_DEBUG "false")
set(HUGR_MLIR_CONFIG_RUSTFLAGS)
set(_cargo_config_dir "${CMAKE_CURRENT_SOURCE_DIR}/../.cargo")
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
