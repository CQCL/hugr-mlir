# This is based on https://michael-f-bryan.github.io/rust-ffi-guide/setting_up.html
set(HUGR_MLIR_CARGO_BUILD_FLAGS "${HUGR_MLIR_CARGO_BUILD_FLAGS}")
set(HUGR_MLIR_CARGO_TEST_FLAGS "${HUGR_MLIR_CARGO_TEST_FLAGS}")
set(HUGR_MLIR_CARGO_DOC_FLAGS "${HUGR_MLIR_CARGO_DOC_FLAGS}")

set(HUGR_MLIR_CONFIG_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target")
set(HUGR_MLIR_CONFIG_PROFILE_RELEASE_DEBUG "false")
set(HUGR_MLIR_CONFIG_RUSTFLAGS "${HUGR_MLIR_CONFIG_RUSTFLAGS}")

# We build the hugr-mlir crate targeting the llvm host triple
set(HUGR_MLIR_CONFIG_TARGET_TRIPLE "${LLVM_HOST_TRIPLE}")

set(_outdir "${CMAKE_CURRENT_BINARY_DIR}/target/${LLVM_HOST_TRIPLE}")
set(_cargo_config_dir "${CMAKE_CURRENT_SOURCE_DIR}/../.cargo")
set(_cargo_config "${_cargo_config_dir}/config.toml")
set(_cargo_lock Cargo.lock)

find_program(CARGO_BINARY cargo REQUIRED)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND _outdir "/debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND HUGR_MLIR_CARGO_BUILD_FLAGS "--release")
    list(APPEND HUGR_MLIR_CARGO_TEST_FLAGS "--release")
    string(APPEND _outdir "/release")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    list(APPEND HUGR_MLIR_CARGO_FLAGS "--release")
    list(APPEND HUGR_MLIR_CARGO_TEST_FLAGS "--release")
    set(HUGR_MLIR_CONFIG_PROFILE_RELEASE_DEBUG "true")
    string(APPEND _outdir "/release")
endif ()


configure_file(
    "${_cargo_config}.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.toml.in"
    @ONLY)

file(GENERATE OUTPUT "${_cargo_config}"
  INPUT "${CMAKE_CURRENT_BINARY_DIR}/config.toml.in"
)

set(_crate_stampfile "${_outdir}/stamp")
set(_crate_depfile "${_outdir}/libhugr_mlir.d")

add_custom_command(
    COMMENT "Compiling hugr-mlir crate"
    OUTPUT "${_crate_stampfile}"
    DEPENDS "${_cargo_config}" HugrMLIR-C
    COMMAND "${CARGO_BINARY}" build ${HUGR_MLIR_CARGO_BUILD_FLAGS}
    COMMAND touch "${_crate_stampfile}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPFILE "${_crate_depfile}"
    BYPRODUCTS "${_cargo_lock}" "${_outdir}/*"
    USES_TERMINAL
)
add_custom_target(hugr_mlir_crate_target ALL
    DEPENDS "${_crate_stampfile}"
)
add_custom_target(hugr_mlir_crate_doc
    COMMENT "Building hugr-mlir crate docs"
    COMMAND "${CARGO_BINARY}" doc
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/target/doc/*"
    USES_TERMINAL
)
add_dependencies(doc hugr_mlir_crate_doc)

add_custom_target(hugr_mlir_crate_test
  COMMENT "Testing hugr-mlir crate"
  DEPENDS "${_cargo_config}" HugrMLIR-C
  COMMAND "${CARGO_BINARY}" test ${HUGR_MLIR_CARGO_TEST_FLAGS}
  USES_TERMINAL
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_dependencies(check hugr_mlir_crate_test)
