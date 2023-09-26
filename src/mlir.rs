use melior::ir::r#type::TypeLike;

/// Generated rust bindings for the definitions in HugrOps.td
/// This feature of melior is not well exercised, so we may well have to turn
/// this off
pub mod hugr {
    use std::{fmt, ops::Deref};

    use hugr::types::FunctionType;
    use melior::ir::{AttributeLike, TypeLike};

    use self::ffi::mlirHugrTypeConstraintAttrGet;

    /// Generated bindings for the C apis of the hugr dialect C++ libraries
    /// The symbols bound all come from libHugrMLIR-C.so
    pub mod ffi {
        include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
    }

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub struct ExtensionAttr<'c> {
        attribute: melior::ir::Attribute<'c>,
    }

    pub fn get_hugr_dialect_handle() -> melior::dialect::DialectHandle {
        unsafe { melior::dialect::DialectHandle::from_raw(ffi::mlirGetDialectHandle__hugr__()) }
    }

    pub fn is_extension_attr(x: melior::ir::Attribute<'_>) -> bool {
        unsafe { ffi::mlirAttributeIsAHugrExtensionAttr(x.to_raw()) }
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

        pub unsafe fn from_raw(attribute: mlir_sys::MlirAttribute) -> Self {
            ExtensionAttr {
                attribute: melior::ir::Attribute::from_raw(attribute),
            }
        }
    }

    impl<'c> From<ExtensionAttr<'c>> for melior::ir::Attribute<'c> {
        fn from(x: ExtensionAttr<'c>) -> Self {
            x.attribute
        }
    }

    impl<'c> TryFrom<melior::ir::Attribute<'c>> for ExtensionAttr<'c> {
        type Error = crate::Error;
        fn try_from(attribute: melior::ir::Attribute<'c>) -> Result<Self, Self::Error> {
            if is_extension_attr(attribute) {
                Ok(ExtensionAttr { attribute })
            } else {
                Err(melior::Error::AttributeExpected(
                    "extension",
                    attribute.to_string(),
                ))?
            }
        }
    }

    impl<'c> fmt::Display for ExtensionAttr<'c> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.attribute, formatter)
        }
    }

    impl<'c> AttributeLike<'c> for ExtensionAttr<'c> {
        fn to_raw(&self) -> mlir_sys::MlirAttribute {
            self.attribute.to_raw()
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub struct ExtensionSetAttr<'c> {
        attribute: melior::ir::Attribute<'c>,
    }

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

        pub unsafe fn from_raw(attribute: mlir_sys::MlirAttribute) -> Self {
            ExtensionSetAttr {
                attribute: melior::ir::Attribute::from_raw(attribute),
            }
        }
    }

    impl<'c> From<ExtensionSetAttr<'c>> for melior::ir::Attribute<'c> {
        fn from(x: ExtensionSetAttr<'c>) -> Self {
            x.attribute
        }
    }

    impl<'c> TryFrom<melior::ir::Attribute<'c>> for ExtensionSetAttr<'c> {
        type Error = crate::Error;
        fn try_from(attribute: melior::ir::Attribute<'c>) -> Result<Self, Self::Error> {
            if is_extension_set_attr(attribute) {
                Ok(ExtensionSetAttr { attribute })
            } else {
                Err(melior::Error::AttributeExpected(
                    "extension set",
                    attribute.to_string(),
                ))?
            }
        }
    }

    impl<'c> fmt::Display for ExtensionSetAttr<'c> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.attribute, formatter)
        }
    }

    impl<'c> AttributeLike<'c> for ExtensionSetAttr<'c> {
        fn to_raw(&self) -> mlir_sys::MlirAttribute {
            self.attribute.to_raw()
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub struct HugrFunctionType<'c> {
        type_: melior::ir::Type<'c>,
    }

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

        pub unsafe fn from_raw(t: mlir_sys::MlirType) -> Self {
            HugrFunctionType {
                type_: melior::ir::Type::from_raw(t),
            }
        }
    }

    impl<'c> From<HugrFunctionType<'c>> for melior::ir::Type<'c> {
        fn from(x: HugrFunctionType<'c>) -> Self {
            x.type_
        }
    }

    impl<'c> TryFrom<melior::ir::Type<'c>> for HugrFunctionType<'c> {
        type Error = crate::Error;
        fn try_from(type_: melior::ir::Type<'c>) -> Result<Self, Self::Error> {
            if is_hugr_function_type(type_) {
                Ok(HugrFunctionType { type_ })
            } else {
                Err(melior::Error::TypeExpected(
                    "hugr function type",
                    type_.to_string(),
                ))?
            }
        }
    }

    impl<'c> fmt::Display for HugrFunctionType<'c> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.type_, formatter)
        }
    }

    impl<'c> TypeLike<'c> for HugrFunctionType<'c> {
        fn to_raw(&self) -> mlir_sys::MlirType {
            self.type_.to_raw()
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub struct SumType<'c> {
        type_: melior::ir::Type<'c>,
    }

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

        pub unsafe fn from_raw(t: mlir_sys::MlirType) -> Self {
            SumType {
                type_: melior::ir::Type::from_raw(t),
            }
        }
    }

    impl<'c> From<SumType<'c>> for melior::ir::Type<'c> {
        fn from(x: SumType<'c>) -> Self {
            x.type_
        }
    }

    impl<'c> TryFrom<melior::ir::Type<'c>> for SumType<'c> {
        type Error = crate::Error;
        fn try_from(type_: melior::ir::Type<'c>) -> Result<Self, Self::Error> {
            if is_sum_type(type_) {
                Ok(SumType { type_ })
            } else {
                Err(melior::Error::TypeExpected("sum type", type_.to_string()))?
            }
        }
    }

    impl<'c> fmt::Display for SumType<'c> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.type_, formatter)
        }
    }

    impl<'c> TypeLike<'c> for SumType<'c> {
        fn to_raw(&self) -> mlir_sys::MlirType {
            self.type_.to_raw()
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub struct SymbolRefAttr<'c> {
        attribute: melior::ir::Attribute<'c>,
    }

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

        pub unsafe fn from_raw(t: mlir_sys::MlirAttribute) -> Self {
            SymbolRefAttr {
                attribute: melior::ir::Attribute::from_raw(t),
            }
        }
    }

    impl<'c> From<SymbolRefAttr<'c>> for melior::ir::Attribute<'c> {
        fn from(x: SymbolRefAttr<'c>) -> Self {
            x.attribute
        }
    }

    impl<'c> From<melior::ir::attribute::FlatSymbolRefAttribute<'c>> for SymbolRefAttr<'c> {
        fn from(x: melior::ir::attribute::FlatSymbolRefAttribute<'c>) -> Self {
            Self {
                attribute: x.into(),
            }
        }
    }

    impl<'c> TryFrom<melior::ir::Attribute<'c>> for SymbolRefAttr<'c> {
        type Error = crate::Error;
        fn try_from(attribute: melior::ir::Attribute<'c>) -> Result<Self, Self::Error> {
            if unsafe { mlir_sys::mlirAttributeIsASymbolRef(attribute.to_raw()) } {
                Ok(SymbolRefAttr { attribute })
            } else {
                Err(melior::Error::AttributeExpected(
                    "sum type",
                    attribute.to_string(),
                ))?
            }
        }
    }

    impl<'c> fmt::Display for SymbolRefAttr<'c> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.attribute, formatter)
        }
    }

    impl<'c> AttributeLike<'c> for SymbolRefAttr<'c> {
        fn to_raw(&self) -> mlir_sys::MlirAttribute {
            self.attribute.to_raw()
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub struct TypeConstraintAttr<'c> {
        attribute: melior::ir::Attribute<'c>,
    }

    impl<'c> TypeConstraintAttr<'c> {
        pub fn new(context: &'c melior::Context, bound: ::hugr::types::TypeBound) -> Self {
            use ::hugr::types::TypeBound;
            let s: melior::StringRef<'_> = match bound {
                TypeBound::Any => "Linear",
                TypeBound::Copyable => "Copyable",
                TypeBound::Eq => "Equatable",
            }
            .into();
            unsafe { Self::from_raw(mlirHugrTypeConstraintAttrGet(context.to_raw(), s.to_raw())) }
        }

        pub unsafe fn from_raw(t: mlir_sys::MlirAttribute) -> Self {
            TypeConstraintAttr {
                attribute: melior::ir::Attribute::from_raw(t),
            }
        }
    }

    impl<'c> From<TypeConstraintAttr<'c>> for melior::ir::Attribute<'c> {
        fn from(x: TypeConstraintAttr<'c>) -> Self {
            x.attribute
        }
    }

    impl<'c> TryFrom<melior::ir::Attribute<'c>> for TypeConstraintAttr<'c> {
        type Error = crate::Error;
        fn try_from(attribute: melior::ir::Attribute<'c>) -> Result<Self, Self::Error> {
            if is_type_constraint_attr(attribute) {
                Ok(TypeConstraintAttr { attribute })
            } else {
                Err(melior::Error::AttributeExpected(
                    "type constraint attr",
                    attribute.to_string(),
                ))?
            }
        }
    }

    impl<'c> fmt::Display for TypeConstraintAttr<'c> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.attribute, formatter)
        }
    }

    impl<'c> AttributeLike<'c> for TypeConstraintAttr<'c> {
        fn to_raw(&self) -> mlir_sys::MlirAttribute {
            self.attribute.to_raw()
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub struct StaticEdgeAttr<'c> {
        attribute: melior::ir::Attribute<'c>,
    }

    impl<'c> StaticEdgeAttr<'c> {
        pub fn new(type_: melior::ir::Type<'c>, sym: SymbolRefAttr<'c>) -> Self {
            unsafe { Self::from_raw(ffi::mlirHugrStaticEdgeAttrGet(type_.to_raw(), sym.to_raw())) }
        }

        pub unsafe fn from_raw(t: mlir_sys::MlirAttribute) -> Self {
            StaticEdgeAttr {
                attribute: melior::ir::Attribute::from_raw(t),
            }
        }
    }

    impl<'c> From<StaticEdgeAttr<'c>> for melior::ir::Attribute<'c> {
        fn from(x: StaticEdgeAttr<'c>) -> Self {
            x.attribute
        }
    }

    impl<'c> TryFrom<melior::ir::Attribute<'c>> for StaticEdgeAttr<'c> {
        type Error = crate::Error;
        fn try_from(attribute: melior::ir::Attribute<'c>) -> Result<Self, Self::Error> {
            if is_static_edge_attr(attribute) {
                Ok(StaticEdgeAttr { attribute })
            } else {
                Err(melior::Error::AttributeExpected(
                    "static edge attr",
                    attribute.to_string(),
                ))?
            }
        }
    }

    impl<'c> fmt::Display for StaticEdgeAttr<'c> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.attribute, formatter)
        }
    }

    impl<'c> AttributeLike<'c> for StaticEdgeAttr<'c> {
        fn to_raw(&self) -> mlir_sys::MlirAttribute {
            self.attribute.to_raw()
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub struct AliasRefType<'c> {
        type_: melior::ir::Type<'c>,
    }

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

        pub unsafe fn from_raw(t: mlir_sys::MlirType) -> Self {
            AliasRefType {
                type_: melior::ir::Type::from_raw(t),
            }
        }
    }

    impl<'c> From<AliasRefType<'c>> for melior::ir::Type<'c> {
        fn from(x: AliasRefType<'c>) -> Self {
            x.type_
        }
    }

    impl<'c> TryFrom<melior::ir::Type<'c>> for AliasRefType<'c> {
        type Error = crate::Error;
        fn try_from(type_: melior::ir::Type<'c>) -> Result<Self, Self::Error> {
            if is_sum_type(type_) {
                Ok(AliasRefType { type_ })
            } else {
                Err(melior::Error::TypeExpected("sum type", type_.to_string()))?
            }
        }
    }

    impl<'c> fmt::Display for AliasRefType<'c> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.type_, formatter)
        }
    }

    impl<'c> TypeLike<'c> for AliasRefType<'c> {
        fn to_raw(&self) -> mlir_sys::MlirType {
            self.type_.to_raw()
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq, Debug)]
    pub struct OpaqueType<'c> {
        type_: melior::ir::Type<'c>,
    }

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

        pub unsafe fn from_raw(t: mlir_sys::MlirType) -> Self {
            OpaqueType {
                type_: melior::ir::Type::from_raw(t),
            }
        }
    }

    impl<'c> From<OpaqueType<'c>> for melior::ir::Type<'c> {
        fn from(x: OpaqueType<'c>) -> Self {
            x.type_
        }
    }

    impl<'c> TryFrom<melior::ir::Type<'c>> for OpaqueType<'c> {
        type Error = crate::Error;
        fn try_from(type_: melior::ir::Type<'c>) -> Result<Self, Self::Error> {
            if is_sum_type(type_) {
                Ok(OpaqueType { type_ })
            } else {
                Err(melior::Error::TypeExpected("sum type", type_.to_string()))?
            }
        }
    }

    impl<'c> fmt::Display for OpaqueType<'c> {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&self.type_, formatter)
        }
    }

    impl<'c> TypeLike<'c> for OpaqueType<'c> {
        fn to_raw(&self) -> mlir_sys::MlirType {
            self.type_.to_raw()
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ModuleOp<'c>(melior::ir::Operation<'c>);

    impl<'c> ModuleOp<'c> {
        pub fn new(body: melior::ir::Region<'c>, loc: melior::ir::Location<'c>) -> Self {
            ModuleOp(
                melior::ir::operation::OperationBuilder::new("hugr.module", loc)
                    .add_regions(vec![body])
                    .build(),
            )
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
        pub fn new<'a>(
            callee: StaticEdgeAttr<'c>,
            inputs: &'_ [melior::ir::Value<'c, 'a>],
            output_types: &'_ [melior::ir::Type<'c>],
            loc: melior::ir::Location<'c>,
        ) -> Self {
            CallOp(
                melior::ir::operation::OperationBuilder::new("hugr.call", loc)
                    .add_attributes(&[(
                        melior::ir::Identifier::new(loc.context().deref(), "callee"),
                        callee.into(),
                    )])
                    .add_operands(inputs)
                    .add_results(output_types)
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
        .into_iter()
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
    pub fn get_test_context() -> melior::Context {
        let dr = melior::dialect::DialectRegistry::new();
        let hugr_dh = super::hugr::get_hugr_dialect_handle();
        hugr_dh.insert_dialect(&dr);
        let ctx = melior::Context::new();
        ctx.append_dialect_registry(&dr);
        ctx.load_all_available_dialects();
        ctx
    }

    #[test]
    fn test_sum_type() {
        let ctx = get_test_context();

        let t_i32 = melior::ir::r#type::IntegerType::new(&ctx, 32);
        let t_i64 = melior::ir::r#type::IntegerType::new(&ctx, 64);
        let sum_type = super::get_sum_type(&ctx, &[t_i32.into(), t_i64.into()]);
        assert_eq!(sum_type.to_string(), "!hugr.sum<i32, i64>");
    }
}
