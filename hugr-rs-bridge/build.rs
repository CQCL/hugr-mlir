fn main() {
    cxx_build::CFG.include_prefix = "hugr-rs-bridge";
    // Note that we don't build the C++ that we generate, instead that is done by cmake
    let _build = cxx_build::bridge("src/lib.rs");

    // TODO there is more to be done to depend on this through rust
}
