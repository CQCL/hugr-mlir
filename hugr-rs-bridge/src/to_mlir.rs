use std::ops::Deref;


fn node_to_op(ctx: melior::ContextRef, hugr: impl hugr::HugrView, n: hugr::Node) -> melior::ir::Operation {
    panic!("unimplemented")

}

pub fn to_mlir(ctx: melior::ContextRef, hugr: impl hugr::HugrView) -> Result<melior::ir::OperationRef, String> {
    let root = hugr.root();
    for c in hugr.children(root) {
        match hugr.get_optype(c) {
            hugr::ops::OpType::Module() => {
                let builder = melior::ir::operation::OperationBuilder::new("hugr.module", melior::ir::Location::unknown(ctx.deref()));
                let region = melior::ir::Region::new();
                let block = melior::ir::Block::new(&[]);
                for c in hugr.children(c) {
                    block.append_operation(node_to_op(ctx, hugr, c));
                }
                builder.add_regions(vec![region]);
                builder.build()
            },

    // Module,
    // FuncDefn,
    // FuncDecl,
    // AliasDecl,
    // AliasDefn,
    // Const,
    // Input,
    // Output,
    // Call,
    // CallIndirect,
    // LoadConstant,
    // DFG,
    // LeafOp,
    // BasicBlock,
    // TailLoop,
    // CFG,
    // Conditional,
    // Case,

        }
        
    }
    panic!("unimplemented")
}
