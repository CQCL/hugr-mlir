use melior::ir::r#type::TypeLike;

#[macro_use]
mod macros;

/// Generated rust bindings for the definitions in HugrOps.td
/// This feature of melior is not well exercised, so we may well have to turn
/// this off
pub mod hugr {
    use std::{any::Any, fmt, ops::Deref};

    use hugr::types::FunctionType;
    use itertools::Itertools;
    use melior::ir::{attribute::StringAttribute, AttributeLike, RegionRef, TypeLike, ValueLike};
    use mlir_sys::mlirAttributeIsASymbolRef;

    use self::ffi::mlirHugrTypeConstraintAttrGet;

    /// Generated bindings for the C apis of the hugr dialect C++ libraries
    /// The symbols bound all come from libHugrMLIR-C.so
    #[allow(non_camel_case_types)]
    pub mod ffi {
        include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
    }

    pub fn get_hugr_dialect_handle() -> melior::dialect::DialectHandle {
        unsafe { melior::dialect::DialectHandle::from_raw(ffi::mlirGetDialectHandle__hugr__()) }
    }

    pub fn is_extension_attr(x: melior::ir::Attribute<'_>) -> bool {
        unsafe { ffi::mlirAttributeIsAHugrExtensionAttr(x.to_raw()) }
    }

    pub fn is_symbol_ref_attr(x: melior::ir::Attribute<'_>) -> bool {
        unsafe { mlirAttributeIsASymbolRef(x.to_raw()) }
    }

    pub fn is_extension_set_attr(x: melior::ir::Attribute<'_>) -> bool {
        unsafe { ffi::mlirAttributeIsAHugrExtensionSetAttr(x.to_raw()) }
    }

    pub fn is_hugr_function_type(x: melior::ir::Type<'_>) -> bool {
        unsafe { ffi::mlirTypeIsAHugrFunctionType(x.to_raw()) }
    }

    pub fn is_sum_type(x: melior::ir::Type<'_>) -> bool {
        unsafe { ffi::mlirTypeIsAHugrSumType(x.to_raw()) }
    }

    pub fn is_type_alias_ref_type(x: melior::ir::Type<'_>) -> bool {
        unsafe { ffi::mlirTypeIsAHugrAliasRefType(x.to_raw()) }
    }

    pub fn is_type_constraint_attr(x: melior::ir::Attribute<'_>) -> bool {
        unsafe { ffi::mlirAttrIsAHugrTypeConstraintAttr(x.to_raw()) }
    }

    pub fn is_static_edge_attr(x: melior::ir::Attribute<'_>) -> bool {
        unsafe { ffi::mlirAttributeIsHugrStaticEdgeAttr(x.to_raw()) }
    }

    pub fn is_sum_attr(x: melior::ir::Attribute<'_>) -> bool {
        unsafe { ffi::mlirAttributeIsHugrSumAttr(x.to_raw()) }
    }

    pub fn is_opaque_type(x: melior::ir::Type<'_>) -> bool {
        unsafe { ffi::mlirTypeIsAHugrOpaqueType(x.to_raw()) }
    }

    declare_attribute!(ExtensionAttr, is_extension_attr, "extension");

    impl<'c> ExtensionAttr<'c> {
        pub fn new<'a>(
            context: &'c melior::Context,
            name: impl Into<melior::StringRef<'a>>,
        ) -> Self {
            unsafe {
                Self::from_raw(ffi::mlirHugrExtensionAttrGet(
                    context.to_raw(),
                    name.into().to_raw(),
                ))
            }
        }
    }

    declare_attribute!(ExtensionSetAttr, is_extension_set_attr, "extension set");

    impl<'c> ExtensionSetAttr<'c> {
        pub fn new(
            context: &'c melior::Context,
            extensions: impl IntoIterator<Item = ExtensionAttr<'c>>,
        ) -> Self {
            let sys_extensions: std::vec::Vec<mlir_sys::MlirAttribute> =
                extensions.into_iter().map(|x| x.to_raw()).collect();
            unsafe {
                Self::from_raw(ffi::mlirHugrExtensionSetAttrGet(
                    context.to_raw(),
                    sys_extensions.len() as i32,
                    sys_extensions.as_ptr(),
                ))
            }
        }
    }

    declare_type!(
        HugrFunctionType,
        is_hugr_function_type,
        "Hugr function type"
    );

    impl<'c> HugrFunctionType<'c> {
        pub fn new(
            extensions: ExtensionSetAttr<'c>,
            function_type: melior::ir::r#type::FunctionType<'c>,
        ) -> Self {
            unsafe {
                Self::from_raw(ffi::mlirHugrFunctionTypeGet(
                    extensions.to_raw(),
                    function_type.to_raw(),
                ))
            }
        }
    }

    declare_type!(SumType, is_sum_type, "Sum type");

    impl<'c> SumType<'c> {
        pub fn new(
            context: &'c melior::Context,
            alts: impl IntoIterator<Item = melior::ir::Type<'c>>,
        ) -> Self {
            let sys_types = alts
                .into_iter()
                .map(|x| x.to_raw())
                .collect::<std::vec::Vec<_>>();
            unsafe {
                Self::from_raw(ffi::mlirHugrSumTypeGet(
                    context.to_raw(),
                    sys_types.len() as i32,
                    sys_types.as_ptr(),
                ))
            }
        }
    }

    declare_attribute!(SymbolRefAttr, is_symbol_ref_attr, "symbol ref attribute");
    impl<'c> SymbolRefAttr<'c> {
        pub fn new<'a, S: Into<melior::StringRef<'a>>>(
            context: &'c melior::Context,
            sym: S,
            nested: impl IntoIterator<Item = melior::ir::attribute::FlatSymbolRefAttribute<'c>>,
        ) -> Self {
            let syms = nested
                .into_iter()
                .map(|x| x.to_raw())
                .collect::<std::vec::Vec<_>>();
            unsafe {
                Self::from_raw(mlir_sys::mlirSymbolRefAttrGet(
                    context.to_raw(),
                    sym.into().to_raw(),
                    syms.len() as isize,
                    syms.as_ptr(),
                ))
            }
        }
    }

    impl<'c> From<melior::ir::attribute::FlatSymbolRefAttribute<'c>> for SymbolRefAttr<'c> {
        fn from(x: melior::ir::attribute::FlatSymbolRefAttribute<'c>) -> Self {
            Self {
                attribute: x.into(),
            }
        }
    }

    declare_attribute!(
        TypeConstraintAttr,
        is_type_constraint_attr,
        "Type constraint attribute"
    );

    impl<'c> TypeConstraintAttr<'c> {
        pub fn new(context: &'c melior::Context, bound: ::hugr::types::TypeBound) -> Self {
            use hugr::types::TypeBound;
            let s: melior::StringRef<'_> = match bound {
                TypeBound::Any => "Linear",
                TypeBound::Copyable => "Copyable",
                TypeBound::Eq => "Equatable",
            }
            .into();
            unsafe { Self::from_raw(mlirHugrTypeConstraintAttrGet(context.to_raw(), s.to_raw())) }
        }
    }

    declare_attribute!(StaticEdgeAttr, is_static_edge_attr, "Static Edge Attribute");
    impl<'c> StaticEdgeAttr<'c> {
        pub fn new(type_: melior::ir::Type<'c>, sym: SymbolRefAttr<'c>) -> Self {
            unsafe { Self::from_raw(ffi::mlirHugrStaticEdgeAttrGet(type_.to_raw(), sym.to_raw())) }
        }
    }

    declare_attribute!(SumAttribute, is_sum_attr, "Sum Attribute");

    impl<'c> SumAttribute<'c> {
        pub fn new(typ: melior::ir::Type<'c>, tag: u32, value: melior::ir::Attribute<'c>) -> Self {
            unsafe { Self::from_raw(ffi::mlirHugrSumAttrGet(typ.to_raw(), tag, value.to_raw())) }
        }
    }

    declare_type!(AliasRefType, is_type_alias_ref_type, "Type Alias Ref Type");

    impl<'c> AliasRefType<'c> {
        pub fn new(
            extensions: ExtensionSetAttr<'c>,
            sym: SymbolRefAttr,
            constraint: TypeConstraintAttr,
        ) -> Self {
            unsafe {
                Self::from_raw(ffi::mlirHugrAliasRefTypeGet(
                    extensions.to_raw(),
                    sym.to_raw(),
                    constraint.to_raw(),
                ))
            }
        }
    }

    declare_type!(OpaqueType, is_opaque_type, "Opaque Type");

    impl<'c> OpaqueType<'c> {
        pub fn new<'a>(
            name: impl Into<melior::StringRef<'a>>,
            extension: ExtensionAttr<'c>,
            args: impl IntoIterator<Item = melior::ir::Attribute<'c>>,
            constraint: TypeConstraintAttr,
        ) -> Self where {
            let sys_args = args.into_iter().map(|x| x.to_raw()).collect::<Vec<_>>();
            unsafe {
                Self::from_raw(ffi::mlirHugrOpaqueTypeGet(
                    name.into().to_raw(),
                    extension.to_raw(),
                    constraint.to_raw(),
                    sys_args.len() as isize,
                    sys_args.as_ptr(),
                ))
            }
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ModuleOp<'c>(melior::ir::Operation<'c>);

    impl<'c> ModuleOp<'c> {
        pub fn new_with_body(body: melior::ir::Region<'c>, loc: melior::ir::Location<'c>) -> Self {
            ModuleOp(
                melior::ir::operation::OperationBuilder::new("hugr.module", loc)
                    .add_regions(vec![body])
                    .build(),
            )
        }
        pub fn new(loc: melior::ir::Location<'c>) -> Self {
            let body = melior::ir::Region::new();
            body.append_block(melior::ir::Block::new(&[]));
            Self::new_with_body(body, loc)
        }

        pub fn body<'a>(&'a self) -> melior::ir::BlockRef<'c, 'a>
        where
            'c: 'a,
        {
            unsafe {
                melior::ir::BlockRef::from_raw(mlir_sys::mlirRegionGetFirstBlock(
                    self.0.region(0).unwrap().clone().to_raw(),
                ))
            }
        }
    }

    impl<'c> From<ModuleOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: ModuleOp<'c>) -> Self {
            op.0
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct OutputOp<'c>(melior::ir::Operation<'c>);

    impl<'c> OutputOp<'c> {
        pub fn new<'a>(
            args: &'_ [melior::ir::Value<'c, 'a>],
            loc: melior::ir::Location<'c>,
        ) -> Self {
            OutputOp(
                melior::ir::operation::OperationBuilder::new("hugr.output", loc)
                    .add_operands(args)
                    .build(),
            )
        }
    }

    impl<'c> From<OutputOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: OutputOp<'c>) -> Self {
            op.0
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct FuncOp<'c>(melior::ir::Operation<'c>);

    impl<'c> FuncOp<'c> {
        pub fn new(
            body: melior::ir::Region<'c>,
            name: melior::ir::attribute::StringAttribute<'c>,
            type_: HugrFunctionType<'c>,
            loc: melior::ir::Location<'c>,
        ) -> Self {
            FuncOp(
                melior::ir::operation::OperationBuilder::new("hugr.func", loc)
                    .add_regions(vec![body])
                    .add_attributes(&[
                        (
                            melior::ir::Identifier::new(name.context().deref(), "sym_name"),
                            name.into(),
                        ),
                        (
                            melior::ir::Identifier::new(name.context().deref(), "function_type"),
                            melior::ir::attribute::TypeAttribute::new(type_.into()).into(),
                        ),
                    ])
                    .build(),
            )
        }

        // pub fn body(&'_ self) -> melior::ir::BlockRef<'c, '_> {
        //     self.0.region(0).unwrap().to_owned().first_block().unwrap()
        // }
    }
    impl<'c> From<FuncOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: FuncOp<'c>) -> Self {
            op.0
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct CallOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<CallOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: CallOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> CallOp<'c> {
        pub fn new_indirect<'a>(
            callee: melior::ir::Value<'c, 'a>,
            inputs: &'_ [melior::ir::Value<'c, 'a>],
            output_types: &'_ [melior::ir::Type<'c>],
            loc: melior::ir::Location<'c>,
        ) -> Self {
            let context = unsafe { loc.context().to_ref() };
            CallOp(
                melior::ir::operation::OperationBuilder::new("hugr.call", loc)
                    .add_attributes(&[(
                        melior::ir::Identifier::new(context, "operand_segment_sizes"),
                        melior::ir::attribute::DenseI32ArrayAttribute::new(
                            context,
                            &[1, inputs.len() as i32],
                        )
                        .into(),
                    )])
                    .add_operands(&[callee])
                    .add_operands(inputs)
                    .add_results(output_types)
                    .build(),
            )
        }
        pub fn new<'a>(
            callee: StaticEdgeAttr<'c>,
            inputs: &'_ [melior::ir::Value<'c, 'a>],
            output_types: &'_ [melior::ir::Type<'c>],
            loc: melior::ir::Location<'c>,
        ) -> Self {
            let context = unsafe { loc.context().to_ref() };
            CallOp(
                melior::ir::operation::OperationBuilder::new("hugr.call", loc)
                    .add_attributes(&[
                        (
                            melior::ir::Identifier::new(loc.context().deref(), "callee_attr"),
                            callee.into(),
                        ),
                        (
                            melior::ir::Identifier::new(context, "operand_segment_sizes"),
                            melior::ir::attribute::DenseI32ArrayAttribute::new(
                                context,
                                &[0, inputs.len() as i32],
                            )
                            .into(),
                        ),
                    ])
                    .add_operands(inputs)
                    .add_results(output_types)
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct SwitchOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<SwitchOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: SwitchOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> SwitchOp<'c> {
        pub fn new(
            inputs: &[melior::ir::Value<'c, '_>],
            successors: &[melior::ir::BlockRef<'c, 'c>],
            loc: melior::ir::Location<'c>,
        ) -> Self {
            use itertools::Itertools;
            SwitchOp(
                melior::ir::operation::OperationBuilder::new("hugr.switch", loc)
                    .add_operands(inputs)
                    .add_successors(
                        successors
                            .iter()
                            .map(|x| x.deref())
                            .collect_vec()
                            .as_slice(),
                    )
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct CfgOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<CfgOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: CfgOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> CfgOp<'c> {
        pub fn new(
            body: melior::ir::Region<'c>,
            output_types: &[melior::ir::Type<'c>],
            inputs: &[melior::ir::Value<'c, '_>],
            loc: melior::ir::Location<'c>,
        ) -> Self {
            CfgOp(
                melior::ir::operation::OperationBuilder::new("hugr.cfg", loc)
                    .add_operands(inputs)
                    .add_results(output_types)
                    .add_regions(vec![body])
                    .build(),
            )
        }

        pub fn body<'a>(&'a self) -> RegionRef<'c, 'a> {
            self.0.region(0).expect("Cfg has exactly one region")
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct MakeTupleOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<MakeTupleOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: MakeTupleOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> MakeTupleOp<'c> {
        pub fn new(
            output_type: melior::ir::Type<'c>,
            inputs: &[melior::ir::Value<'c, '_>],
            loc: melior::ir::Location<'c>,
        ) -> Self {
            MakeTupleOp(
                melior::ir::operation::OperationBuilder::new("hugr.make_tuple", loc)
                    .add_operands(inputs)
                    .add_results(vec![output_type].as_slice())
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ConstOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<ConstOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: ConstOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> ConstOp<'c> {
        pub fn new(
            name: impl Into<melior::ir::Attribute<'c>>,
            typ: impl Into<melior::ir::Type<'c>>,
            value: impl Into<melior::ir::Attribute<'c>>,
            loc: melior::ir::Location<'c>,
        ) -> Self {
            let context = unsafe { loc.context().to_ref() };
            ConstOp(
                melior::ir::operation::OperationBuilder::new("hugr.const", loc)
                    .add_attributes(&[
                        (
                            melior::ir::Identifier::new(context, "sym_name"),
                            name.into(),
                        ),
                        (
                            melior::ir::Identifier::new(context, "type"),
                            melior::ir::attribute::TypeAttribute::new(typ.into()).into(),
                        ),
                        (melior::ir::Identifier::new(context, "value"), value.into()),
                    ])
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct TagOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<TagOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: TagOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> TagOp<'c> {
        pub fn new(
            result_type: melior::ir::Type<'c>,
            tag: u32,
            inputs: &[melior::ir::Value<'c, '_>],
            loc: melior::ir::Location<'c>,
        ) -> Self {
            let context = unsafe { loc.context().to_ref() };
            TagOp(
                melior::ir::operation::OperationBuilder::new("hugr.tag", loc)
                    .add_attributes(&[(
                        melior::ir::Identifier::new(context, "tag"),
                        melior::ir::attribute::IntegerAttribute::new(
                            tag as i64,
                            melior::ir::r#type::IntegerType::new(context, 32).into(),
                        )
                        .into(),
                    )])
                    .add_results(&[result_type])
                    .add_operands(inputs)
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct LoadConstantOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<LoadConstantOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: LoadConstantOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> LoadConstantOp<'c> {
        pub fn new(
            result_type: impl Into<melior::ir::Type<'c>>,
            target: impl Into<melior::ir::Attribute<'c>>,
            loc: melior::ir::Location<'c>,
        ) -> Self {
            let context = unsafe { loc.context().to_ref() };

            LoadConstantOp(
                melior::ir::operation::OperationBuilder::new("hugr.load_constant", loc)
                    .add_attributes(&[(
                        melior::ir::Identifier::new(context, "const_ref"),
                        target.into(),
                    )])
                    .add_results(&[result_type.into()])
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ExtensionOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<ExtensionOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: ExtensionOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> ExtensionOp<'c> {
        pub fn new(
            result_types: &[melior::ir::Type<'c>],
            name: impl AsRef<str>,
            extension_set: ExtensionSetAttr<'c>,
            args: &[melior::ir::Value<'c, '_>],
            loc: melior::ir::Location<'c>,
        ) -> Self {
            let context = unsafe { loc.context().to_ref() };

            ExtensionOp(
                melior::ir::operation::OperationBuilder::new("hugr.ext_op", loc)
                    .add_attributes(&[
                        (
                            melior::ir::Identifier::new(context, "extensions"),
                            extension_set.into(),
                        ),
                        (
                            melior::ir::Identifier::new(context, "hugr_opname"),
                            melior::ir::attribute::StringAttribute::new(context, name.as_ref())
                                .into(),
                        ),
                    ])
                    .add_results(result_types)
                    .add_operands(args)
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ConditionalOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<ConditionalOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: ConditionalOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> ConditionalOp<'c> {
        pub fn new(
            result_types: &[melior::ir::Type<'c>],
            args: &[melior::ir::Value<'c, '_>],
            cases: impl IntoIterator<Item = melior::ir::Region<'c>>,
            loc: melior::ir::Location<'c>,
        ) -> Self {
            ConditionalOp(
                melior::ir::operation::OperationBuilder::new("hugr.conditional", loc)
                    .add_results(result_types)
                    .add_operands(args)
                    .add_regions(cases.into_iter().collect())
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct UnpackTupleOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<UnpackTupleOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: UnpackTupleOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> UnpackTupleOp<'c> {
        pub fn new(
            result_types: &[melior::ir::Type<'c>],
            arg: melior::ir::Value<'c, '_>,
            loc: melior::ir::Location<'c>,
        ) -> Self {
            UnpackTupleOp(
                melior::ir::operation::OperationBuilder::new("hugr.unpack_tuple", loc)
                    .add_results(result_types)
                    .add_operands(&[arg])
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct TailLoopOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<TailLoopOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: TailLoopOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> TailLoopOp<'c> {
        pub fn new(
            outputs_types: &[melior::ir::Type<'c>],
            inputs: &[melior::ir::Value<'c, '_>],
            passthrough_inputs: &[melior::ir::Value<'c, '_>],
            body: melior::ir::Region<'c>,
            loc: melior::ir::Location<'c>,
        ) -> Self {
            let context = unsafe { loc.context().to_ref() };
            let inputs_types = inputs.iter().map(|x| x.r#type()).collect_vec();
            let passthrough_inputs_types =
                passthrough_inputs.iter().map(|x| x.r#type()).collect_vec();
            let predicate_type = SumType::new(
                context,
                [
                    melior::ir::r#type::TupleType::new(context, &inputs_types).into(),
                    melior::ir::r#type::TupleType::new(context, outputs_types).into(),
                ],
            );

            TailLoopOp(
                melior::ir::operation::OperationBuilder::new("hugr.tailloop", loc)
                    .add_results(outputs_types)
                    .add_results(passthrough_inputs_types.as_slice())
                    .add_operands(inputs)
                    .add_operands(passthrough_inputs)
                    .add_attributes(&[
                        (
                            melior::ir::Identifier::new(context, "operand_segment_sizes"),
                            melior::ir::attribute::DenseI32ArrayAttribute::new(
                                context,
                                &[inputs.len() as i32, passthrough_inputs.len() as i32],
                            )
                            .into(),
                        ),
                        (
                            melior::ir::Identifier::new(context, "result_segment_sizes"),
                            melior::ir::attribute::DenseI32ArrayAttribute::new(
                                context,
                                &[outputs_types.len() as i32, passthrough_inputs.len() as i32],
                            )
                            .into(),
                        ),
                        (
                            melior::ir::Identifier::new(context, "predicate_type"),
                            melior::ir::attribute::TypeAttribute::new(predicate_type.into()).into(),
                        ),
                    ])
                    .add_regions(vec![body])
                    .build(),
            )
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct LiftOp<'c>(melior::ir::Operation<'c>);

    impl<'c> From<LiftOp<'c>> for melior::ir::Operation<'c> {
        fn from(op: LiftOp<'c>) -> Self {
            op.0
        }
    }

    impl<'c> LiftOp<'c> {
        pub fn new(
            outputs_types: &[melior::ir::Type<'c>],
            inputs: &[melior::ir::Value<'c, '_>],
            extension: ExtensionAttr<'c>,
            loc: melior::ir::Location<'c>,
        ) -> Self {
            let context = unsafe { loc.context().to_ref() };
            LiftOp(
                melior::ir::operation::OperationBuilder::new("hugr.lift", loc)
                    .add_results(outputs_types)
                    .add_operands(inputs)
                    .add_attributes(&[(
                        melior::ir::Identifier::new(context, "extensions"),
                        ExtensionSetAttr::new(context, [extension]).into(),
                    )])
                    .build(),
            )
        }
    }
}

pub fn get_sum_type<'a>(
    context: &'a melior::Context,
    components: &[melior::ir::Type<'a>],
) -> melior::ir::Type<'a> {
    let mut unwrapped_components = components
        .iter()
        .map(|x| x.to_raw())
        .collect::<std::vec::Vec<_>>();
    unsafe {
        melior::ir::Type::from_raw(hugr::ffi::mlirHugrSumTypeGet(
            context.to_raw(),
            unwrapped_components.len() as i32,
            unwrapped_components.as_mut_ptr(),
        ))
    }
}

#[cfg(test)]
pub mod test {
    use rstest::{fixture, rstest};
    #[fixture]
    pub fn test_context() -> melior::Context {
        let dr = melior::dialect::DialectRegistry::new();
        let hugr_dh = super::hugr::get_hugr_dialect_handle();
        hugr_dh.insert_dialect(&dr);
        let ctx = melior::Context::new();
        ctx.append_dialect_registry(&dr);
        ctx.load_all_available_dialects();
        ctx
    }

    #[rstest]
    fn test_sum_type(test_context: melior::Context) {
        let t_i32 = melior::ir::r#type::IntegerType::new(&test_context, 32);
        let t_i64 = melior::ir::r#type::IntegerType::new(&test_context, 64);
        let sum_type = super::get_sum_type(&test_context, &[t_i32.into(), t_i64.into()]);
        assert_eq!(sum_type.to_string(), "!hugr.sum<i32, i64>");
    }
}
