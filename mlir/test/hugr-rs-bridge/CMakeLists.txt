add_executable(test-hugr-rs-bridge
  TestHugrRsBridge.cpp
)

target_link_libraries(test-hugr-rs-bridge
  PRIVATE
  hugr-rs-bridge
)
set_target_properties(test-hugr-rs-bridge
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

# TODO We don't want to install tests in general. We do it here so that we can
# verify that tests work when run from the install tree
install(TARGETS test-hugr-rs-bridge
  TYPE RUNTIME
)
