use std::borrow::Borrow;
use crate::{mlir, Error};

pub fn extension_id_to_extension_attr<'c>(
    context: &'c melior::Context,
    id: &'_ hugr::extension::ExtensionId,
) -> mlir::hugr::ExtensionAttr<'c> {
    crate::mlir::hugr::ExtensionAttr::new(context, id.borrow())
}

pub fn extension_set_to_extension_set_attr<'c>(
    context: &'c melior::Context,
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
    ctx: &'c melior::Context,
    type_: &'_ hugr::types::Type,
) -> Result<melior::ir::Type<'c>, Error> {
    use hugr::types::{PrimType, TypeEnum};
    match type_.clone().into() {
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
            collect_type_row::<Vec<_>>(ctx, &row)?.as_slice(),
        )
        .into()),
        TypeEnum::Prim(p) => match p {
            PrimType::Extension(custom_type) => Ok(crate::mlir::hugr::OpaqueType::new(
                custom_type.name().as_ref(),
                extension_id_to_extension_attr(ctx, custom_type.extension()),
                collect_type_args::<Vec<_>>(ctx, custom_type.args())?,
                mlir::hugr::TypeConstraintAttr::new(ctx, custom_type.bound()),
            )
            .into()),
            PrimType::Alias(alias_decl) => Ok(mlir::hugr::AliasRefType::new(
                mlir::hugr::ExtensionSetAttr::new(ctx, []),
                melior::ir::attribute::FlatSymbolRefAttribute::new(ctx, alias_decl.name.as_ref())
                    .into(),
                mlir::hugr::TypeConstraintAttr::new(ctx, alias_decl.bound),
            )
            .into()),
            PrimType::Function(function_type) => {
                Ok(hugr_to_mlir_function_type(ctx, &function_type)?.into())
            }
        },
    }
}

pub fn hugr_to_mlir_function_type<'a>(
    ctx: &'a melior::Context,
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

pub fn collect_type_row<'c, R: FromIterator<melior::ir::Type<'c>>>(
    context: &'c melior::Context,
    row: &'_ hugr::types::TypeRow,
) -> Result<R, Error> {
    row.iter().map(|x| hugr_to_mlir_type(context, x)).collect()
}

pub fn collect_type_args<'c, 'a, R: FromIterator<melior::ir::Attribute<'c>>>(
    ctx: &'c melior::Context,
    args: impl IntoIterator<Item = &'a hugr::types::TypeArg>,
) -> Result<R, Error> {
    use hugr::types::TypeArg;
    args.into_iter()
        .map(|x| match x {
            &TypeArg::Type { ref ty } => {
                Ok(melior::ir::attribute::TypeAttribute::new(hugr_to_mlir_type(ctx, ty)?).into())
            }
            &TypeArg::BoundedNat { n } => Ok(melior::ir::attribute::IntegerAttribute::new(
                n as i64,
                melior::ir::r#type::IntegerType::new(ctx, 64).into(),
            )
            .into()),
            &TypeArg::Opaque { .. } => todo!(),
            &TypeArg::Sequence { ref args } => Ok(melior::ir::attribute::ArrayAttribute::new(
                ctx,
                &collect_type_args::<Vec<_>>(ctx, args)?,
            )
            .into()),
            &TypeArg::Extensions { ref es } => {
                Ok(extension_set_to_extension_set_attr(ctx, es).into())
            }
            ta => panic!("unimplemented type_arg: {:?}", ta),
        })
        .collect::<Result<R, Error>>()
}
