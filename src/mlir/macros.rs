// these macros are largely liberated from private macros in melior

macro_rules! declare_attribute {
    ($name: ident, $is_type: ident, $string: expr) => {
        #[derive(Clone, Copy, Eq, PartialEq, Debug)]
        pub struct $name<'c> {
            attribute: melior::ir::Attribute<'c>,
        }
        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: mlir_sys::MlirAttribute) -> Self {
                Self {
                    attribute: melior::ir::attribute::Attribute::from_raw(raw),
                }
            }
        }

        impl<'c> From<$name<'c>> for melior::ir::Attribute<'c> {
            fn from(x: $name<'c>) -> Self {
                x.attribute
            }
        }

        impl<'c> TryFrom<melior::ir::attribute::Attribute<'c>> for $name<'c> {
            type Error = $crate::Error;

            fn try_from(
                attribute: melior::ir::attribute::Attribute<'c>,
            ) -> Result<Self, Self::Error> {
                if $is_type(attribute) {
                    Ok(unsafe { Self::from_raw(attribute.to_raw()) })
                } else {
                    Err(melior::Error::AttributeExpected($string, attribute.to_string()).into())
                }
            }
        }

        impl<'c> melior::ir::attribute::AttributeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_sys::MlirAttribute {
                self.attribute.to_raw()
            }
        }

        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.attribute, formatter)
            }
        }
    };
}

macro_rules! declare_type {
    ($name: ident, $is_type: ident, $string: expr) => {
        #[derive(Clone, Copy, Eq, PartialEq, Debug)]
        pub struct $name<'c> {
            type_: melior::ir::Type<'c>,
        }

        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: mlir_sys::MlirType) -> Self {
                Self {
                    type_: melior::ir::Type::from_raw(raw),
                }
            }
        }

        impl<'c> From<$name<'c>> for melior::ir::Type<'c> {
            fn from(x: $name<'c>) -> Self {
                x.type_
            }
        }

        impl<'c> TryFrom<melior::ir::Type<'c>> for $name<'c> {
            type Error = $crate::Error;

            fn try_from(t: melior::ir::Type<'c>) -> Result<Self, Self::Error> {
                if $is_type(t) {
                    Ok(unsafe { Self::from_raw(t.to_raw()) })
                } else {
                    Err(melior::Error::TypeExpected($string, t.to_string()).into())
                }
            }
        }

        impl<'c> melior::ir::r#type::TypeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_sys::MlirType {
                self.type_.to_raw()
            }
        }

        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.type_, formatter)
            }
        }
    };
}

macro_rules! declare_op {
    ($name: ident, $opname: expr) => {
        #[repr(transparent)]
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct $name<'c>(melior::ir::Operation<'c>);

        impl<'c> From<$name<'c>> for melior::ir::Operation<'c> {
            fn from(op: $name<'c>) -> Self {
                op.0
            }
        }

        impl<'c,'a> TryFrom<&'a melior::ir::Operation<'c>> for &'a $name<'c> {
            type Error = ();
            fn try_from(op: &'a melior::ir::Operation<'c>) -> Result<Self,Self::Error> {
                let ctx = unsafe { op.context().to_ref() };
                if op.name() == $name::opname(&ctx) {
                    Ok(unsafe { std::mem::transmute(op) })
                } else {
                    Err(())
                }
            }
        }

        impl<'c> $name<'c> {
            pub fn opname(ctx: &'c melior::Context) -> melior::ir::Identifier<'c> {
                melior::ir::Identifier::new(ctx, $opname)
            }

            pub fn builder(loc: melior::ir::Location<'c>) -> melior::ir::operation::OperationBuilder {
                let ctx = unsafe { loc.context().to_ref() };
                melior::ir::operation::OperationBuilder::new(Self::opname(ctx).as_string_ref().as_str().unwrap(), loc)
            }
        }
    };
}
