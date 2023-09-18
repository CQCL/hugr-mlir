set(HUGR_MLIR_CARGO_FLAGS "${HUGR_MLIR_CARGO_FLAGS}")

# This is based on https://michael-f-bryan.github.io/rust-ffi-guide/setting_up.html
set(HUGR_MLIR_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target")
set(_cargo_config_dir "${CMAKE_CURRENT_SOURCE_DIR}/../.cargo")
set(_cargo_config "${_cargo_config_dir}/config.toml")
set(_cargo_lock "${CMAKE_CURRENT_SOURCE_DIR}/../Cargo.lock")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_outdir "${HUGR_MLIR_TARGET_DIR}/debug")
else ()
    list(APPEND HUGR_MLIR_CARGO_FLAGS "--release")
    set(_outdir "${HUGR_MLIR_TARGET_DIR}/release")
endif ()

set(_libhugr_rs_bridge_a "${_outdir}/libhugr_rs_bridge.a")
set(_libhugr_rs_bridge_a_d "${_outdir}/libhugr_rs_bridge.d")

if(DEFINED ENV{LIBCLANG_PATH})
    message(STATUS "LIBCLANG_PATH: $ENV{LIBCLANG_PATH}")
else()
    message(SEND_ERROR "The environment variable LIBCLANG_PATH is not set. This is required for bindgen")
endif()

set(BINDGEN_EXTRA_CLANG_ARGS)
# This allows bindgen, as called from within, mlir-sys to find the mlir headers
# when building against a build tree
if(MLIR_INCLUDE_DIRS)
    foreach(_d ${MLIR_INCLUDE_DIRS})
        string(APPEND BINDGEN_EXTRA_CLANG_ARGS " -I${_d}")
    endforeach()
endif()

# get_target_property(HUGR_MLIR_MLIR_INCLUDE_DIRS mlir-headers-interface INTERFACE_INCLUDE_DIRECTORIES)

configure_file(
    "${_cargo_config}.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.toml.in"
    @ONLY)

file(GENERATE OUTPUT "${_cargo_config}"
  INPUT "${CMAKE_CURRENT_BINARY_DIR}/config.toml.in"
)

add_custom_command(
    COMMENT "Compiling hugr_rs_bridge rust crate"
    OUTPUT
    "${_libhugr_rs_bridge_a}"
    "${HUGR_MLIR_TARGET_DIR}/cxxbridge/hugr-rs-bridge/src/lib.rs.cc"
    "${HUGR_MLIR_TARGET_DIR}/cxxbridge/hugr-rs-bridge/src/lib.rs.h"
    "${HUGR_MLIR_TARGET_DIR}/cxxbridge/rust/cxx.h"
    DEPENDS "${_cargo_config}"
    COMMAND cargo build ${HUGR_MLIR_CARGO_FLAGS}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPFILE "${_libhugr_rs_bridge_a_d}"
    BYPRODUCTS "${_libhugr_rs_bridge_a_d}" "${_cargo_lock}"
    USES_TERMINAL
)

# depending on files in cmake is flaky, depending on them through a custom
# target works reliably
add_custom_target(hugr-rs-bridge-cargo
    DEPENDS
    "${_libhugr_rs_bridge_a}"
    "${HUGR_MLIR_TARGET_DIR}/cxxbridge/hugr-rs-bridge/src/lib.rs.cc"
    "${HUGR_MLIR_TARGET_DIR}/cxxbridge/hugr-rs-bridge/src/lib.rs.h"
    "${HUGR_MLIR_TARGET_DIR}/cxxbridge/rust/cxx.h"
)

add_library(hugr-rs-bridge
    "${HUGR_MLIR_TARGET_DIR}/cxxbridge/hugr-rs-bridge/src/lib.rs.cc"
    cxx/HugrRsBridge.cpp
)
add_dependencies(hugr-rs-bridge hugr-rs-bridge-cargo)
target_link_libraries(hugr-rs-bridge PUBLIC
  "${_libhugr_rs_bridge_a}"
  mlir-headers-interface
  MLIRSupport
)
target_include_directories(hugr-rs-bridge PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${HUGR_MLIR_TARGET_DIR}/cxxbridge>"
)
llvm_update_compile_flags(hugr-rs-bridge)
mlir_check_all_link_libraries(hugr-rs-bridge)

add_custom_target(hugr-rs-bridge-doc
    COMMAND cargo doc
    DEPENDS "${_cargo_config}"
    COMMENT "Building hugr-rs-bridge docs"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)

# For now we can't run any tests  in the hugr-rs-bridge crate beacuse linking
# add_custom_target(hugr-rs-bridge-test
#     COMMAND cargo test
#     COMMENT "Running hugr-rs-bridge tests"
#     WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#     USES_TERMINAL
# )
# add_dependencies(check hugr-rs-bridge-test)

add_dependencies(doc hugr-rs-bridge-doc)
install(DIRECTORY "${HUGR_MLIR_TARGET_DIR}/docs"
  TYPE DOC
)
