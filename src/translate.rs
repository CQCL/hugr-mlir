use std::borrow::Borrow;
use std::env;
use std::vec::Vec;

use crate::hugr_to_mlir::hugr_to_mlir;
use crate::mlir::emit_error;
use crate::mlir_to_hugr::mlir_to_hugr;
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
            emit_error(loc, e.to_string());
            mlir_sys::MlirOperation {
                ptr: std::ptr::null_mut(),
            }
        }
    }
}

pub fn translate_mlir_to_hugr<'c, E: Into<crate::Error>>(
    op: mlir_sys::MlirOperation,
    go: impl FnOnce(&hugr::Hugr) -> Result<(), E>,
) -> mlir_sys::MlirLogicalResult {
    unsafe {
        let op1 = melior::ir::OperationRef::from_raw(op);
        if let Err(e) = mlir_to_hugr(&op1).and_then(|x| go(&x).map_err(Into::into)) {
            emit_error(op1.loc(), e.to_string());
            mlir_sys::mlirLogicalResultFailure()
        } else {
            mlir_sys::mlirLogicalResultSuccess()
        }
    }
}

mod ffi {
    use mlir_sys::MlirStringRef;

    use crate::{mlir, mlir_to_hugr};

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

    pub extern "C" fn translate_mlir_to_hugr_rmp(
        op: mlir_sys::MlirOperation,
        emit_context: *const mlir::hugr::ffi::EmitContext
    ) -> mlir_sys::MlirLogicalResult {
        super::translate_mlir_to_hugr(op, |x|
            rmp_serde::to_vec(x).map(|y| mlir::emit_stringref((&emit_context).into(), y))
        )
    }

    pub extern "C" fn translate_mlir_to_hugr_json(
        op: mlir_sys::MlirOperation,
        emit_context: *const mlir::hugr::ffi::EmitContext
    ) -> mlir_sys::MlirLogicalResult {
        super::translate_mlir_to_hugr(op, |x|
            serde_json::to_vec(x).map(|y| mlir::emit_stringref((&emit_context).into(), y))
        )
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

        mlir::hugr::ffi::mlirHugrRegisterTranslationFromMLIR(
            melior::StringRef::from("mlir-to-hugr-json").to_raw(),
            melior::StringRef::from("mlir to hugr translation").to_raw(),
            Some(ffi::translate_mlir_to_hugr_json),
            Some(ffi::register_translation_dialects),
        );

        mlir::hugr::ffi::mlirHugrRegisterTranslationFromMLIR(
            melior::StringRef::from("mlir-to-hugr-rmp").to_raw(),
            melior::StringRef::from("mlir to hugr translation").to_raw(),
            Some(ffi::translate_mlir_to_hugr_rmp),
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
