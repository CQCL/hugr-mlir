use crate::mlir::hugr::ffi::mlirHugrTranslateStringRefToMLIRFunction;
use crate::{mlir, Error, Result};
use hugr::hugr::IdentList;
use melior::Context;
use std::borrow::Borrow;
use std::ops::Deref;

pub fn extension_id_to_extension_attr<'c>(
    context: &'c Context,
    id: &'_ hugr::extension::ExtensionId,
) -> mlir::hugr::ExtensionAttr<'c> {
    crate::mlir::hugr::ExtensionAttr::new(context, id.borrow())
}

pub fn extension_set_to_extension_set_attr<'c>(
    context: &'c Context,
    set: &'_ hugr::extension::ExtensionSet,
) -> mlir::hugr::ExtensionSetAttr<'c> {
    crate::mlir::hugr::ExtensionSetAttr::new(
        context,
        set.iter()
            .map(|x| extension_id_to_extension_attr(context, x))
            .collect::<Vec<_>>(),
    )
}

pub fn hugr_to_mlir_type<'c>(
    ctx: &'c Context,
    type_: &hugr::types::Type,
) -> Result<melior::ir::Type<'c>, Error> {
    use hugr::types::TypeEnum;
    match type_.as_type_enum() {
        TypeEnum::Sum(sum_type) => {
            let mut alts = std::vec::Vec::<_>::new();
            for i in 0usize.. {
                if let Some(t) = sum_type.get_variant(i) {
                    alts.push(hugr_to_mlir_type(ctx, t)?)
                } else {
                    break;
                }
            }
            Ok(mlir::hugr::SumType::new(ctx, alts).into())
        }
        TypeEnum::Tuple(row) => Ok(melior::ir::r#type::TupleType::new(
            ctx,
            collect_type_row::<Vec<_>>(ctx, row)?.as_slice(),
        )
        .into()),
        TypeEnum::Extension(ref custom_type) => {
            let mb_t = match custom_type.extension().deref() {
                "prelude" => match custom_type.name().as_str() {
                    "usize" => Some(unsafe {
                        melior::ir::Type::from_raw(mlir_sys::mlirIndexTypeGet(ctx.to_raw()))
                    }),
                    _ => None
                },
                "arithmetic.int.types" => match custom_type.name().as_str() {
                    "int" => {
                        assert!(!custom_type.args().is_empty());
                        if let hugr::types::type_param::TypeArg::BoundedNat { n } =
                            custom_type.args()[0]
                        {
                            Some(melior::ir::r#type::IntegerType::new(ctx, n as u32).into())
                        } else {
                            panic!(
                                "int type does not have a bounded nat arg: {:?}",
                                custom_type
                            )
                        }
                    }
                    _ => None
                },
                "arithmetic.float.types" => match custom_type.name().as_str() {
                    "float64" => {
                        assert!(custom_type.args().is_empty());
                        Some(unsafe {
                            melior::ir::Type::from_raw(mlir_sys::mlirF64TypeGet(ctx.to_raw()))
                        })
                    }
                    "float32" => {
                        assert!(custom_type.args().is_empty());
                        Some(unsafe {
                            melior::ir::Type::from_raw(mlir_sys::mlirF32TypeGet(ctx.to_raw()))
                        })
                    }
                    _ => None
                },
                _ => None
            };
            if let Some(t) = mb_t {
                Ok(t)
            } else {
                Ok(crate::mlir::hugr::OpaqueType::new(
                    custom_type.name().as_ref(),
                    extension_id_to_extension_attr(ctx, custom_type.extension()),
                    collect_type_args::<Vec<_>>(ctx, custom_type.args())?,
                    mlir::hugr::TypeConstraintAttr::new(ctx, custom_type.bound()),
                    ).into())
            }
        },
        TypeEnum::Alias(ref alias_decl) => Ok(mlir::hugr::AliasRefType::new(
            mlir::hugr::ExtensionSetAttr::new(ctx, []),
            melior::ir::attribute::FlatSymbolRefAttribute::new(ctx, alias_decl.name.as_ref())
                .into(),
            mlir::hugr::TypeConstraintAttr::new(ctx, alias_decl.bound),
        )
        .into()),
        TypeEnum::Function(ref function_type) if function_type.params().is_empty() => {
            Ok(hugr_to_mlir_function_type(ctx, function_type.body())?.into())
        }
        TypeEnum::Function(ref _function_type) => {
            panic!("unimplemented: TypeEnum::Function with params")
        }
        TypeEnum::Variable(_size, _bound) => panic!("unimplemented: TypeEnum::Variable"),
    }
}

pub fn hugr_to_mlir_function_type<'a>(
    ctx: &'a Context,
    type_: &'_ hugr::types::FunctionType,
) -> Result<mlir::hugr::HugrFunctionType<'a>, Error> {
    let type_extensions = extension_set_to_extension_set_attr(ctx, &type_.extension_reqs);
    let type_inputs = collect_type_row::<Vec<_>>(ctx, &type_.input)?;
    let type_outputs = collect_type_row::<Vec<_>>(ctx, &type_.output)?;
    Ok(mlir::hugr::HugrFunctionType::new(
        type_extensions,
        melior::ir::r#type::FunctionType::new(ctx, type_inputs.as_slice(), type_outputs.as_slice()),
    ))
}

pub fn collect_type_row_vec<'c>(
    context: &'c Context,
    row: &'_ hugr::types::TypeRow,
) -> Result<Vec<melior::ir::Type<'c>>> {
    collect_type_row(context, row)
}

pub fn collect_type_row<'c, R: FromIterator<melior::ir::Type<'c>>>(
    context: &'c Context,
    row: &'_ hugr::types::TypeRow,
) -> Result<R, Error> {
    row.iter().map(|x| hugr_to_mlir_type(context, x)).collect()
}

pub fn collect_type_args<'c, 'a, R: FromIterator<melior::ir::Attribute<'c>>>(
    ctx: &'c Context,
    args: impl IntoIterator<Item = &'a hugr::types::TypeArg>,
) -> Result<R, Error> {
    use hugr::types::TypeArg;
    args.into_iter()
        .map(|x| match x {
            TypeArg::Type { ty } => {
                Ok(melior::ir::attribute::TypeAttribute::new(hugr_to_mlir_type(ctx, ty)?).into())
            }
            &TypeArg::BoundedNat { n } => Ok(melior::ir::attribute::IntegerAttribute::new(
                n as i64,
                melior::ir::r#type::IntegerType::new(ctx, 64).into(),
            )
            .into()),
            &TypeArg::Opaque { .. } => todo!(),
            TypeArg::Sequence { ref elems } => Ok(melior::ir::attribute::ArrayAttribute::new(
                ctx,
                &collect_type_args::<Vec<_>>(ctx, elems)?,
            )
            .into()),
            TypeArg::Extensions { es } => Ok(extension_set_to_extension_set_attr(ctx, es).into()),
            ta => panic!("unimplemented type_arg: {:?}", ta),
        })
        .collect::<Result<R, Error>>()
}
