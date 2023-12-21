use std::collections::HashMap;

use hugr::hugr::hugrmut::HugrMut;

use crate::mlir;
use crate::Result;
use hugr::Hugr;

pub mod hash;

// trait MlirToHugr {
//     fn go(&self, hugr: &HugrMut)

// }
//

// struct TranslationState<'a,'b, B: 'b + hugr::builder::Container> where
//     'a: 'b
// {
//     builder: B,
//     scope: HashMap<hash::HashableValue<'a,'b>, (hugr::Node,hugr::OutgoingPort)>
// }

// trait EmitHugr {
//     type Builder: hugr::builder::Container;

//     fn emit(&self, builder: &mut Self::Builder) -> Result<()> {
//         todo!()
//     }
// }

// impl EmitHugr for mlir::hugr::ModuleOp<'_> {

// }

// impl<'a,'b,B: 'b + hugr::Builder::Container> TranslationState<'a,'b,B> {
//     fn new(builder: B) -> Self {
//         Self {
//             builder,
//             scope: HashMap::new()
//         }
//     }

//     fn module(op: &'b mlir::hugr::ModuleOp<'a>) -> Result<Hugr> {
//         let mut mod_builder = hugr::builder::ModuleBuilder::new();
//         mod_builder.add_alias_declare(…)

//     }

//     fn container(&mut self,
//         _builder: &mut impl hugr::builder::Container,
//         _region: impl IntoIterator<Item = melior::ir::OperationRef<'a, 'b>>,
//     ) -> Result<()> {
//         todo!()
//     }
// }

// fn block_ops<'c, 'b>(
//     block: &'b melior::ir::Block<'c>,
// ) -> impl Iterator<Item = melior::ir::OperationRef<'c, 'b>> {
//     std::iter::successors(block.first_operation(), |x| {
//         unsafe { x.to_ref() }.next_in_block()
//     })
// }

pub fn mlir_to_hugr(op: &mlir::hugr::ModuleOp<'_>) -> Result<hugr::Hugr> {
    // let mut state = TranslationState::new();
    // if let Ok(op1) = TryInto::<&mlir::hugr::ModuleOp>::try_into(op) {
    //     let mut b = hugr::builder::ModuleBuilder::new();
    //     build_container(&mut b, block_ops(&op1.body()))?;
    // }
    todo!()
}
