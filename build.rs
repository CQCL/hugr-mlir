use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let include_dirs = std::env::var("DEP_MLIR_INCLUDE_DIRS")?;
    let bindings = bindgen::builder()
        .clang_args(include_dirs.split(';').map(|x| format!("-I{}", x)))
        .header("mlir/include/hugr-mlir-c/Dialects.h")
        .header("mlir/include/hugr-mlir-c/Translate.h")
        .allowlist_file(".*mlir/include/hugr-mlir-c/(Translate|Dialects).h")
        .allowlist_recursively(false)
        .raw_line("use mlir_sys::*;")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo:rerun-if-changed=mlir/include/hugr-mlir-c/Dialects.h");
    println!("cargo:rerun-if-changed=mlir/include/hugr-mlir-c/Translate.h");
    println!("cargo:rerun-if-env-changed=DEP_MLIR_INCLUDE_DIRS");
    println!("cargo:rerun-if-env-changed=OUT_DIR");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_DIR");
    Ok(())
}
