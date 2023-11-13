use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let include_dirs = std::env::var("DEP_MLIR_INCLUDE_DIRS")?;
    let llvm_config_path = std::env::var("DEP_MLIR_CONFIG_PATH")?;
    let cflags = llvm_config(&llvm_config_path, "--cflags")?;
    let bindings = bindgen::builder()
        .clang_args(include_dirs.split(';').map(|x| format!("-I{}", x)))
        .clang_args(cflags.split(' '))
        .header("mlir/include/hugr-mlir-c/Dialects.h")
        .header("mlir/include/hugr-mlir-c/Translate.h")
        .header("mlir/include/hugr-mlir-c/Analysis.h")
        .allowlist_file("(.*/mlir/include/hugr-mlir-c/((Translate|Dialects|Analysis)\\.h)|.*/hugr-mlir/.*/Passes\\.capi\\.h\\.inc)")
        .allowlist_recursively(false)
        .raw_line("use mlir_sys::*;")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo:rerun-if-env-changed=DEP_MLIR_INCLUDE_DIRS");
    println!("cargo:rerun-if-env-changed=OUT_DIR");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_DIR");
    Ok(())
}

fn llvm_config(llvm_config_path: &str, argument: &str) -> Result<String, Box<dyn std::error::Error>> {
    let prefix = std::path::Path::new(llvm_config_path).join("bin");
    let call = format!(
        "{} --link-shared {}",
        prefix.join("llvm-config").display(),
        argument
    );

    Ok(std::str::from_utf8(
        &if cfg!(target_os = "windows") {
            std::process::Command::new("cmd").args(["/C", &call]).output()?
        } else {
            std::process::Command::new("sh").arg("-c").arg(&call).output()?
        }
        .stdout,
    )?
    .trim()
    .to_string())
}
