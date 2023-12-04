use crate::{mlir, Error, Result};
use anyhow::anyhow;

use itertools::zip_eq;
use melior::Context;

use super::types::hugr_to_mlir_type;

pub fn hugr_to_mlir_value<'c>(
    context: &'c Context,
    typ: &hugr::types::Type,
    value: &hugr::values::Value,
) -> Result<melior::ir::Attribute<'c>> {
    use downcast_rs::Downcast;
    use hugr::types::TypeEnum;
    use hugr::values::Value;
    match value {
        Value::Tuple { vs } => {
            let TypeEnum::Tuple(ref typerow) = typ.as_type_enum() else { Err(anyhow!("not a tuple type"))? };
            Ok(melior::ir::attribute::ArrayAttribute::new(
                context,
                zip_eq(typerow.iter(), vs.iter())
                    .map(|(ty, x)| hugr_to_mlir_value(context, ty, x))
                    .collect::<Result<Vec<_>>>()?
                    .as_slice(),
            )
            .into())
        }
        &Value::Sum { tag, ref value } => {
            let TypeEnum::Sum(ref sum_type) =  typ.as_type_enum() else { Err(anyhow!("not a sum type"))? };
            let Some(variant_type) = sum_type.get_variant(tag) else { Err(anyhow!("bad tag for sum type"))? };
            Ok(mlir::hugr::SumAttribute::new(
                hugr_to_mlir_type(context, typ)?,
                tag as u32,
                hugr_to_mlir_value(context, variant_type, value)?,
            )
            .into())
        }
        &Value::Extension {ref c } => {
            if let Some(i) =
                c.0.downcast_ref::<hugr::std_extensions::arithmetic::int_types::ConstIntS>()
            {
                Ok(melior::ir::attribute::IntegerAttribute::new(
                    i.value(),
                    melior::ir::r#type::IntegerType::new(context, i.log_width().into()).into(),
                )
                .into())
            } else if let Some(i) =
                c.0.downcast_ref::<hugr::std_extensions::arithmetic::int_types::ConstIntU>()
            {
                // this as i64 is naughty
                Ok(melior::ir::attribute::IntegerAttribute::new(
                    i.value() as i64,
                    melior::ir::r#type::IntegerType::new(context, i.log_width().into()).into(),
                )
                .into())
            } else if let Some(f) =
                c.0.downcast_ref::<hugr::std_extensions::arithmetic::float_types::ConstF64>()
            {
                Ok(
                    melior::ir::attribute::FloatAttribute::new(context, f.value(), unsafe {
                        melior::ir::Type::from_raw(mlir_sys::mlirF64TypeGet(context.to_raw()))
                    })
                    .into(),
                )
            } else if let Some(f) = c.0.downcast_ref::<hugr::extension::prelude::ConstUsize>() {
                Ok(
                    melior::ir::attribute::IntegerAttribute::new(f.value() as i64, unsafe {
                        melior::ir::Type::from_raw(mlir_sys::mlirIndexTypeGet(context.to_raw()))
                    })
                    .into(),
                )
            } else {
                Err(anyhow!(
                    "hugr_to_mlir_value:unimplemented extension constant: {:?}",
                    c
                ))
            }
        }
        x => Err(anyhow!("Unimplemented hugr_to_mlir_value: {:?}", x)),
    }
}
