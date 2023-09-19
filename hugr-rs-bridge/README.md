# hugr-rs-bridge

A crate to facilitate interoperation between `hugr`, the mlir crates `mlir-sys` and `melior`, and `hugr-mlir`. We use the `cxx` crate to generate rust/c++ interop code, as well as importing the hugr-mlir CAPI.

Initially this crate will hold the implementation of mappings `hugr <-> mlir`. It may also grow features to allow `hugr-mlir` to call into the `hugr` type system for validation and inference.

Part of a cargo workspace rooted in the root of the repository.

We produce a static library `libhugr_rs_bridge.a`, a c++ source file and several headers. 

Note that we cannot, at present, write tests or binaries in this crate because
we do not configure cargo to link to either HugrMLIRDialectCAPI or the generated
c++. Doing so would require:
 * Configuring multiple libraries via environment variables to link in build.rs
 * Building the generated c++ in build.rs, likely having to manually propagate
   ad-hoc configuration from cmake into the `cc` crate.
   
This crate is tested instead in /mlir/tests/test-hugr-rs-bridge;

The cmake file:
 * At configuration time we generate `.cargo/config.toml` in the root of the repo, where we set environment variables to configure our dependencies to use the configured tree for mlir. We configure the target directory to be inside the cmake build tree.
 * We define a `hugr-rs-bridge` target that builds the generated c++, links the staticlib output of the rust crate, and includes the generated headers. The rest of the system depends on the rust crate only through `hugr-rs-bridge`.
 * We add `cargo doc` to the `doc` target.





