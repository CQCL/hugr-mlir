use std::path::PathBuf;

fn main() {
    let mut cmake = cmake::Config::new("mlir");
    let mut clang_args = std::vec::Vec::new();
    if let Ok(mlir_path) = std::env::var("DEP_MLIR_CONFIG_PATH") {
        cmake.define("MLIR_DIR", format!("{}/lib/cmake/mlir", mlir_path));
        clang_args.push(format!("-I{}/include", mlir_path));
    }
    let dst = cmake.generator("Ninja").build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=HugrMLIR-C");

    let bindings = bindgen::Builder::default()
        .clang_args(clang_args)
        .header("mlir/include/hugr-mlir-c/Dialects.h")
        .allowlist_file("mlir/include/hugr-mlir-c/Dialects.h")
        .allowlist_recursively(false)
        .raw_line("use mlir_sys::*;")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo:rerun-if-changed=./mlir");
    println!("cargo:rerun-if-env-changed=DEP_MLIR_CONFIG_PATH");

    let target_dir =
        std::path::PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or("target".to_string()));
    let symlink = target_dir.join("hugr-mlir_out");
    if !symlink.exists() || symlink.is_symlink() {
        if symlink.exists() {
            std::fs::remove_file(symlink.clone()).expect("Failed to remove old symlink");
        }
        std::os::unix::fs::symlink(out_path.clone(), symlink).expect("Failed to create symlink");
    }

    let cc_source = out_path.join("build").join("compile_commands.json");
    let cc_sl = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("mlir")
        .join("compile_commands.json");
    if cc_source.exists() && (!cc_sl.exists() || cc_sl.is_symlink()) {
        if cc_sl.exists() {
            std::fs::remove_file(cc_sl.clone())
                .expect("Failed to remove compile_commands.json symlink");
        }
        if cc_sl.exists() {
            panic!("impossible");
        }
        std::os::unix::fs::symlink(cc_source.clone(), cc_sl.clone()).expect(&format!(
            "Failed to create compile_commands.json symlink: {:?}, {:?}",
            &cc_source, &cc_sl
        ));
    }
}
