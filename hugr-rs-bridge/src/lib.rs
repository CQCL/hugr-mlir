pub struct Hugr(hugr::Hugr);

#[cxx::bridge(namespace = "hugr_rs_bridge::detail")]
pub mod ffi {
    struct ParseResult {
        msg: String,
        result: *mut Hugr,
    }

    struct SerializeToStringResult {
        msg: String,
        result: String,
    }

    struct SerializeToBytesResult {
        msg: String,
        result: *mut u8,
        size: usize,
    }

    struct OperationResult {
        msg: String,
        result: *mut Operation,
    }

    extern "Rust" {
        type Hugr;
        pub fn get_example_hugr() -> *mut Hugr;
        pub fn parse_hugr_json(buf: &[u8]) -> ParseResult;
        pub fn parse_hugr_rmp(buf: &[u8]) -> ParseResult;
        pub unsafe fn hugr_to_mlir(hugr: Box<Hugr>, context: *mut MLIRContext) -> OperationResult;

        pub fn hugr_to_json(hugr: &Hugr) -> SerializeToStringResult;
        pub fn hugr_to_rmp(hugr: &Hugr) -> SerializeToBytesResult;
    }

    unsafe extern "C++" {
        include!("mlir/IR/MLIRContext.h");
        include!("mlir/IR/Operation.h");
        #[namespace = "mlir"]
        type MLIRContext;
        #[namespace = "mlir"]
        type Operation;
    }
}

fn try_deserialize_hugr<'a, E: std::fmt::Display>(
    buf: &'a [u8],
    parse: impl FnOnce(&'a [u8]) -> Result<hugr::Hugr, E>,
) -> ffi::ParseResult {
    match parse(buf) {
        Err(e) => ffi::ParseResult {
            msg: e.to_string(),
            result: std::ptr::null_mut(),
        },
        Ok(h) => ffi::ParseResult {
            msg: "".into(),
            result: Box::into_raw(Box::new(Hugr(h))),
        },
    }
}

pub fn parse_hugr_json(buf: &[u8]) -> ffi::ParseResult {
    try_deserialize_hugr(buf, serde_json::from_slice::<hugr::Hugr>)
}

pub fn parse_hugr_rmp(buf: &[u8]) -> ffi::ParseResult {
    try_deserialize_hugr(buf, rmp_serde::from_slice::<hugr::Hugr>)
}

pub unsafe fn hugr_to_mlir(
    _hugr: Box<Hugr>,
    _context: *mut ffi::MLIRContext,
) -> ffi::OperationResult {
    panic!("unimplemented")
}

pub fn hugr_to_json(h: &Hugr) -> ffi::SerializeToStringResult {
    match serde_json::to_string_pretty(&h.0) {
        Err(e) => ffi::SerializeToStringResult {
            msg: e.to_string(),
            result: "".into(),
        },
        Ok(result) => ffi::SerializeToStringResult {
            msg: "".into(),
            result,
        },
    }
}

pub fn hugr_to_rmp(h: &Hugr) -> ffi::SerializeToBytesResult {
    match rmp_serde::to_vec(&h.0) {
        Err(e) => ffi::SerializeToBytesResult {
            msg: e.to_string(),
            result: std::ptr::null_mut(),
            size: 0,
        },
        Ok(result) => {
            let size = result.len();
            let mut b = result.into_boxed_slice();
            let result = b.as_mut_ptr();
            std::mem::forget(b);
            ffi::SerializeToBytesResult {
                msg: "".into(),
                result,
                size,
            }
        }
    }
}

fn get_example_hugr_impl() -> Result<hugr::Hugr, hugr::builder::BuildError> {
    use hugr::builder::{Dataflow, DataflowSubContainer, HugrBuilder};
    const NAT: hugr::types::Type = hugr::extension::prelude::USIZE_T;
    let mut module_builder = hugr::builder::ModuleBuilder::new();

    let f_id = module_builder.declare(
        "main",
        hugr::types::FunctionType::new(hugr::type_row![NAT], hugr::type_row![NAT]).pure(),
    )?;

    let mut f_build = module_builder.define_declaration(&f_id)?;
    let call = f_build.call(&f_id, f_build.input_wires())?;

    f_build.finish_with_outputs(call.outputs())?;
    module_builder.finish_prelude_hugr().map_err(|x| x.into())
}

pub fn get_example_hugr() -> *mut Hugr {
    match get_example_hugr_impl() {
        Err(_) => std::ptr::null_mut(),
        Ok(x) => Box::into_raw(Box::new(Hugr(x))),
    }
}

// These come from HugrMLIRDialectCAPI, to which, at present, we do not link.
// This choice means that we can't write tests that call these functions here.
extern "C" {
    pub fn mlirHugrTypeConstraintAttrGet(
        context: mlir_sys::MlirContext,
        kind: *const i8,
    ) -> mlir_sys::MlirAttribute;
}

// #[cfg(test)]
// mod test {
//     #[test]
//     fn can_get_type_constraint_attr() {
//         let context = melior::Context::new();
//         let kind = std::ffi::CString::new("Linear").unwrap();
//         let attr = unsafe {melior::ir::Attribute::from_raw(crate::mlirHugrTypeConstraintAttrGet(context.to_raw(), kind.into_raw())) };

//     }
// }