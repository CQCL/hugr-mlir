
#[derive(PartialEq,Eq,Debug)]
struct HashableOperationRef<'a,'b> (melior::ir::OperationRef<'a,'b>);

impl<'a,'b> std::hash::Hash for HashableOperationRef<'a,'b> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use melior::ir::ValueLike;
        state.write_u32(unsafe { crate::mlir::hugr::ffi::mlirOperationHash(self.0.to_raw()) })
    }
}

#[derive(PartialEq,Eq,Debug)]
struct HashableType<'a> (melior::ir::Type<'a>);

impl<'a> std::hash::Hash for HashableType<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use melior::ir::TypeLike;
        state.write_u32(unsafe { crate::mlir::hugr::ffi::mlirTypeHash(self.0.to_raw()) })
    }
}

#[derive(PartialEq,Eq,Debug)]
struct HashableAttribute<'a> (melior::ir::Attribute<'a>);

impl<'a> std::hash::Hash for HashableAttribute<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use melior::ir::AttributeLike;
        state.write_u32(unsafe { crate::mlir::hugr::ffi::mlirAttributeHash(self.0.to_raw()) })
    }
}

#[derive(PartialEq,Eq,Debug)]
struct HashableBlockRef<'a,'b> (melior::ir::BlockRef<'a,'b>);

impl<'a,'b> std::hash::Hash for HashableBlockRef<'a,'b> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(unsafe { crate::mlir::hugr::ffi::mlirBlockHash(self.0.to_raw()) })
    }
}

#[derive(PartialEq,Eq,Debug)]
struct HashableRegionRef<'a,'b> (melior::ir::RegionRef<'a,'b>);

impl<'a,'b> std::hash::Hash for HashableRegionRef<'a,'b> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(unsafe { crate::mlir::hugr::ffi::mlirRegionHash(self.0.to_raw()) })
    }
}

#[derive(PartialEq,Eq,Debug)]
struct HashableValue<'a,'b> (melior::ir::Value<'a,'b>);

impl<'a,'b> std::hash::Hash for HashableValue<'a,'b> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use melior::ir::ValueLike;
        state.write_u32(unsafe { crate::mlir::hugr::ffi::mlirValueHash(self.0.to_raw()) })
    }
}
