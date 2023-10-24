use std::borrow::Borrow;
use std::env;
use std::vec::Vec;

use crate::hugr_to_mlir::hugr_to_mlir;
use crate::{mlir, Error, Result};

pub fn translate_hugr_to_mlir<'c, E: Into<crate::Error>>(
    src: &[u8],
    loc: melior::ir::Location<'c>,
    go: impl FnOnce(&[u8]) -> Result<hugr::Hugr, E>,
) -> Result<melior::ir::Module<'c>> where
{
    let hugr = go(src).map_err(Into::into)?;
    hugr_to_mlir(loc, &hugr)
}

fn translate_hugr_raw_to_mlir(
    raw_src: mlir_sys::MlirStringRef,
    raw_loc: mlir_sys::MlirLocation,
    go: impl FnOnce(&[u8]) -> Result<hugr::Hugr, Error>,
) -> mlir_sys::MlirOperation {
    let loc = unsafe { melior::ir::Location::from_raw(raw_loc) };
    // dbg!("translate::translate_hugr_raw_to_mlir: {:?}", loc);

    let src = unsafe { melior::StringRef::from_raw(raw_src) };

    match translate_hugr_to_mlir(src.as_bytes(), loc, go) {
        Ok(module) => unsafe { mlir_sys::mlirModuleGetOperation(module.into_raw()) },
        Err(e) => {
            unsafe {
                mlir_sys::mlirEmitError(
                    raw_loc,
                    std::ffi::CString::new(e.to_string())
                        .unwrap_or(std::ffi::CString::from_vec_unchecked(
                            "CString nul error".into(),
                        ))
                        .as_bytes_with_nul()
                        .as_ptr() as *const i8,
                )
            };
            mlir_sys::MlirOperation {
                ptr: std::ptr::null_mut(),
            }
        }
    }
}

mod ffi {
    use crate::mlir;

    pub extern "C" fn translate_hugr_json_to_mlir(
        raw_src: mlir_sys::MlirStringRef,
        raw_loc: mlir_sys::MlirLocation,
    ) -> mlir_sys::MlirOperation {
        super::translate_hugr_raw_to_mlir(raw_src, raw_loc, |src| {
            serde_json::from_slice(src).map_err(Into::into)
        })
    }
    pub extern "C" fn translate_hugr_rmp_to_mlir(
        raw_src: mlir_sys::MlirStringRef,
        raw_loc: mlir_sys::MlirLocation,
    ) -> mlir_sys::MlirOperation {
        super::translate_hugr_raw_to_mlir(raw_src, raw_loc, |src| {
            rmp_serde::from_slice(src).map_err(Into::into)
        })
    }

    pub extern "C" fn register_translation_dialects(registry: mlir_sys::MlirDialectRegistry) {
        let reg = unsafe { melior::dialect::DialectRegistry::from_raw(registry) };
        mlir::hugr::get_hugr_dialect_handle().insert_dialect(&reg);
        std::mem::forget(reg);
    }
}

fn translate_main(args: &[String]) -> Result<(), String> {
    // one imagines that there must be a better way...
    let args: Vec<std::ffi::CString> = args
        .iter()
        .map(|x| std::ffi::CString::new(x.as_str()).map_err(|x| x.to_string()))
        .collect::<Result<_, _>>()?;
    let argsv: Vec<*const std::os::raw::c_char> = args
        .iter()
        .map(|x| x.as_bytes_with_nul().as_ptr() as *const std::os::raw::c_char)
        .collect();

    // dbg!("translate::translate_main args: {:?}", &args);

    unsafe {
        mlir::hugr::ffi::mlirHugrRegisterTranslationToMLIR(
            melior::StringRef::from("hugr-json-to-mlir").to_raw(),
            melior::StringRef::from("hugr to mlir translation").to_raw(),
            Some(ffi::translate_hugr_json_to_mlir),
            Some(ffi::register_translation_dialects),
        );

        mlir::hugr::ffi::mlirHugrRegisterTranslationToMLIR(
            melior::StringRef::from("hugr-rmp-to-mlir").to_raw(),
            melior::StringRef::from("hugr to mlir translation").to_raw(),
            Some(ffi::translate_hugr_rmp_to_mlir),
            Some(ffi::register_translation_dialects),
        );
    }

    match unsafe {
        crate::mlir::hugr::ffi::mlirHugrTranslateMain(args.len() as i32, argsv.as_ptr())
    } {
        0 => Ok(()),
        x => Err(format!("failure: {}", x)),
    }
}

pub fn main() {
    std::process::exit(
        match translate_main(env::args().collect::<Vec<_>>().as_slice()) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("{}", e);
                1
            }
        },
    );
}
