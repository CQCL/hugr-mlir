use anyhow::anyhow;
use hugr::ops::OpTrait;
use itertools::{zip_eq, Itertools};
use melior::ir::attribute::FlatSymbolRefAttribute;
use melior::ir::{BlockRef, Type};
use std::borrow::{BorrowMut, Cow};
use std::iter::{empty, zip};
use std::ops::Deref;

use hugr::HugrView;
use hugr::{ops::BasicBlock, types::EdgeKind};
use melior::{
    ir::{Block, Location, Operation, OperationRef, Region, Value},
    Context,
};
use std::collections::{HashMap, HashSet};
use std::vec::Vec;

use crate::mlir::hugr::StaticEdgeAttr;
use crate::{mlir, Error, Result};

pub mod types;
use types::*;

pub mod value;
use value::*;

type Scope<'a, 'b> = HashMap<(hugr::Node, hugr::Port), Value<'a, 'b>>;
type ScopeItem<'a, 'b> = <Scope<'a, 'b> as IntoIterator>::Item;

type SymbolItem<'a> = (
    mlir::hugr::StaticEdgeAttr<'a>,
    Type<'a>,
    melior::ir::attribute::StringAttribute<'a>,
);

struct Symboliser<'a, V: HugrView> {
    context: &'a Context,
    hugr: &'a V,
    allocated_symbols: HashSet<String>,
    node_to_symbol: HashMap<hugr::Node, SymbolItem<'a>>,
    next_unique: i32,
}

impl<'a, V: HugrView> Clone for Symboliser<'a, V> {
    fn clone(&self) -> Self {
        Self {
            context: self.context,
            hugr: self.hugr,
            allocated_symbols: self.allocated_symbols.clone(),
            node_to_symbol: self.node_to_symbol.clone(),
            next_unique: self.next_unique,
        }
    }
}

impl<'a, V: HugrView> Symboliser<'a, V> {
    fn new(context: &'a Context, hugr: &'a V) -> Self {
        Self {
            context,
            hugr,
            allocated_symbols: HashSet::new(),
            node_to_symbol: HashMap::new(),
            next_unique: 0,
        }
    }

    fn get_or_alloc<'b>(&'b mut self, k: hugr::Node) -> Result<SymbolItem<'a>> {
        if let Some(r) = self.node_to_symbol.get(&k) {
            Ok(*r)
        } else {
            use hugr::ops::OpType;
            use hugr::NodeIndex;
            let (ty, mut sym) = match self.hugr.get_optype(k) {
                &OpType::FuncDecl(hugr::ops::FuncDecl {
                    ref name,
                    ref signature,
                })
                | &OpType::FuncDefn(hugr::ops::FuncDefn {
                    ref name,
                    ref signature,
                }) => {
                    let ty = hugr_to_mlir_function_type(self.context, signature)?.into();
                    Ok((ty, name.to_string()))
                }
                OpType::Const(const_) => {
                    let ty = hugr_to_mlir_type(self.context, const_.const_type())?;
                    Ok((ty, format!("const_{}", k.index())))
                }
                opty => Err(anyhow!("Bad optype for static edge: {:?}", opty)),
            }?;
            if self.allocated_symbols.contains(&sym) {
                sym = format!("{}_{}", sym, self.next_unique);
                self.next_unique += 1;
            }
            let attr = mlir::hugr::StaticEdgeAttr::new(
                ty,
                mlir::hugr::SymbolRefAttr::new(self.context, sym.as_ref(), empty()),
            );
            let None = self.node_to_symbol.insert(k, (attr, ty, melior::ir::attribute::StringAttribute::new(self.context, &sym))) else {
                panic!("Expected node to be unmapped: {:?}", k);
            };
            if !self.allocated_symbols.insert(sym.clone()) {
                panic!("Expected sym to be unallocated: {}", sym);
            }
            Ok(*self.node_to_symbol.get(&k).unwrap())
        }
    }
}

trait EmitMlir {
    type Op<'a>: Into<Operation<'a>>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>>;
}

impl EmitMlir for hugr::ops::OpType {
    type Op<'a> = Operation<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::OpType;
        match self {
            OpType::Conditional(ref conditional) => conditional.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::Output(ref output) => output.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::FuncDecl(ref funcdecl) => funcdecl.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::FuncDefn(ref funcdefn) => funcdefn.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::Module(ref module) => module.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::Call(ref call) => call.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::CallIndirect(ref callindirect) => callindirect.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::CFG(ref cfg) => cfg.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::LeafOp(ref leaf) => leaf.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::Const(ref const_) => const_.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::LoadConstant(ref lc) => lc.emit(state, node, result_types, inputs, loc).map(Into::into),
            OpType::TailLoop(ref tailloop) => tailloop.emit(state, node, result_types, inputs, loc).map(Into::into),
            _ => todo!()
        }
    }
}

impl EmitMlir for hugr::ops::Conditional {
    type Op<'a> = mlir::hugr::ConditionalOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        let cases = state
            .hugr
            .children(node)
            .map(|case_n| {
                let r: Region<'a> = Region::new();
                let b = Block::new(&[]);
                state.build_dataflow_block(case_n, &b)?;
                r.append_block(b);
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(mlir::hugr::ConditionalOp::new(
            result_types,
            inputs,
            cases,
            loc,
        ))
    }
}

impl EmitMlir for hugr::ops::Output {
    type Op<'a> = mlir::hugr::OutputOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        _state: &mut TranslationState<'a, 'b, V>,
        _node: hugr::Node,
        _result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        Ok(mlir::hugr::OutputOp::new(inputs, loc))
    }
}

impl EmitMlir for hugr::ops::FuncDefn {
    type Op<'a> = mlir::hugr::FuncOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        _result_types: &[Type<'a>],
        _inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        let (_, ty, sym) = state.symbols.get_or_alloc(node)?;
        let block = Block::new(&[]);
        state.build_dataflow_block(node, &block)?;
        let body = Region::new();
        body.append_block(block);
        Ok(mlir::hugr::FuncOp::new(body, sym, ty.try_into()?,loc))
    }
}

impl EmitMlir for hugr::ops::FuncDecl {
    type Op<'a> = mlir::hugr::FuncOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        _result_types: &[Type<'a>],
        _inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        let (_, ty, sym) = state.symbols.get_or_alloc(node)?;
        let body = Region::new();
        Ok(mlir::hugr::FuncOp::new(body, sym, ty.try_into()?,loc))
    }
}

impl EmitMlir for hugr::ops::Module {
    type Op<'a> = mlir::hugr::ModuleOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        _result_types: &[Type<'a>],
        _inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        let body = Region::new();
        let block = Block::new(&[]);
        state.push_block(&block, empty(), |mut state| {
            for c in state.hugr.children(node) {
                state.node_to_op(c, loc)?;
            }
            Ok::<_,crate::Error>(())
        })?;
        body.append_block(block);
        Ok(mlir::hugr::ModuleOp::new_with_body(body, loc))
    }
}

impl EmitMlir for hugr::ops::Call {
    type Op<'a> = mlir::hugr::CallOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::OpType;
        use hugr::PortIndex;
        let static_index = self.signature.input.len();
        let static_port = state
            .hugr
            .node_inputs(node)
            .find(|p| p.index() == static_index)
            .ok_or(anyhow!("Failed to find static edge to function"))?;
        assert_eq!(
            state.hugr.get_optype(node).port_kind(static_port),
            Some(EdgeKind::Static(hugr::types::Type::new_function(
                self.signature.clone()
            )))
        );
        let (target_n, _) = state
            .hugr
            .linked_ports(node, static_port)
            .collect_vec()
            .into_iter()
            .exactly_one()?;
        let (static_edge, _, _) = state.symbols.get_or_alloc(target_n)?;

        Ok(mlir::hugr::CallOp::new(static_edge, inputs, result_types, loc))
    }
}

impl EmitMlir for hugr::ops::CallIndirect {
    type Op<'a> = mlir::hugr::CallOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        Ok(mlir::hugr::CallOp::new_indirect(inputs[0], &inputs[1..], result_types, loc))
    }
}

impl EmitMlir for hugr::ops::CFG {
    type Op<'a> = mlir::hugr::CfgOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::controlflow::BasicBlock;
        use hugr::ops::OpType;
        use hugr::types::EdgeKind;
        use hugr::PortIndex;
        let body: Region<'a> = Region::new();
        {
            let mut node_to_block = HashMap::<hugr::Node, BlockRef<'a, '_>>::new();
            for c in state.hugr.children(node) {
                let b = body.append_block(melior::ir::Block::new(&[]));
                node_to_block.insert(c, b);
            }
            for (dfb_node, block) in node_to_block.iter() {
                match state.hugr.get_optype(*dfb_node) {
                    optype @ &OpType::BasicBlock(BasicBlock::DFB { .. }) => state
                        .build_dataflow_block_term(*dfb_node, block.deref(), |state, inputs, _| {
                            let successors = state
                                .hugr
                                .node_outputs(*dfb_node)
                                .filter(|p| {
                                    optype.port_kind(*p) == Some(hugr::types::EdgeKind::ControlFlow)
                                })
                                .map(|p| {
                                    Ok(node_to_block[&state
                                        .hugr
                                        .linked_ports(*dfb_node, p)
                                        .collect_vec()
                                        .into_iter()
                                        .exactly_one()?
                                        .0])
                                })
                                .collect::<Result<Vec<_>>>()?;
                            Ok((
                                (),
                                mlir::hugr::SwitchOp::new(inputs, &successors, loc).into(),
                            ))
                        })?,
                    &OpType::BasicBlock(BasicBlock::Exit { ref cfg_outputs }) => {
                        let args = collect_type_row_vec(state.context, cfg_outputs)?
                            .into_iter()
                            .map(|t| block.add_argument(t, loc))
                            .collect_vec();
                        block.append_operation(mlir::hugr::OutputOp::new(&args, loc).into());
                    }
                    &_ => Err(anyhow!("not a basic block"))?,
                };
            }
        }
        Ok(mlir::hugr::CfgOp::new(body, result_types, inputs, loc))
    }
}

impl EmitMlir for hugr::ops::custom::ExternalOp {
    type Op<'a> = mlir::hugr::ExtensionOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        _node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::custom::ExternalOp;
        let name = match self {
            ExternalOp::Extension(ref e) => e.def().name(),
            ExternalOp::Opaque(ref o) => o.name(),
        };
        let extensions = extension_set_to_extension_set_attr(
            state.context,
            &self.signature().extension_reqs,
        );
        Ok(mlir::hugr::ExtensionOp::new(
            result_types,
            name,
            extensions,
            inputs,
            loc,
        ))
    }
}

impl EmitMlir for hugr::ops::LeafOp {
    type Op<'a> = melior::ir::Operation<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::LeafOp;
        match self {
            LeafOp::CustomOp(custom) => custom.emit(state, node, result_types, inputs, loc).map(Into::into),
            LeafOp::MakeTuple { .. } => {
                assert_eq!(result_types.len(), 1);
                Ok(mlir::hugr::MakeTupleOp::new(result_types[0], inputs, loc).into())
            }
            LeafOp::UnpackTuple { .. } => {
                assert_eq!(inputs.len(), 1);
                Ok(mlir::hugr::UnpackTupleOp::new(result_types, inputs[0], loc).into())
            }
            LeafOp::Tag { tag, .. } => {
                assert_eq!(result_types.len(), 1);
                Ok(mlir::hugr::TagOp::new(result_types[0], *tag as u32, inputs, loc).into())
            }
            LeafOp::Lift { new_extension, .. } => {
                Ok(mlir::hugr::LiftOp::new(result_types, inputs, extension_id_to_extension_attr(state.context, new_extension),loc).into())
            }
            &_ => panic!("Unimplemented leafop: {:?}", self)
        }
    }
}

impl EmitMlir for hugr::ops::Const {
    type Op<'a> = mlir::hugr::ConstOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        let (_, _, name) = state.symbols.get_or_alloc(node)?;
        let ty = hugr_to_mlir_type(state.context, self.const_type())?;
        let val = hugr_to_mlir_value(state.context, self.const_type(), self.value())?;
        Ok(mlir::hugr::ConstOp::new(name, ty, val, loc))
    }
}

impl EmitMlir for hugr::ops::LoadConstant {
    type Op<'a> = mlir::hugr::LoadConstantOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        use hugr::PortIndex;
        let static_index = 0usize;
        let static_port = state
            .hugr
            .node_inputs(node)
            .find(|p| p.index() == static_index)
            .ok_or(anyhow!("Failed to find static edge to function"))?;
        let (target_n, _) = state
            .hugr
            .linked_ports(node, static_port)
            .collect_vec()
            .into_iter()
            .exactly_one()?;
        let (edge, _, _) = state.symbols.get_or_alloc(target_n)?;
        assert_eq!(result_types.len(), 1);
        Ok(mlir::hugr::LoadConstantOp::new(result_types[0], edge, loc))
    }
}

impl EmitMlir for hugr::ops::TailLoop {
    type Op<'a> = mlir::hugr::TailLoopOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        node: hugr::Node,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        assert_eq!(
            result_types.len(),
            self.just_outputs.len() + self.rest.len()
        );
        let init_inputs = &inputs[0..self.just_inputs.len()];
        let passthrough_inputs = &inputs[self.just_inputs.len()..];
        let outputs_types = &result_types[0..self.just_outputs.len()];
        let block = Block::new(&[]);
        state.build_dataflow_block_term(node, &block, |mut state, outputs, out_n| {
            state.hugr.get_optype(out_n).emit(&mut state, out_n, &[], outputs, loc).map(|x|((),x))
        })?;
        let body = Region::new();
        body.append_block(block);
        Ok(mlir::hugr::TailLoopOp::new(outputs_types, inputs, passthrough_inputs, body, loc))
    }
}

struct TranslationState<'a, 'b, V: HugrView>
where
    'a: 'b,
{
    context: &'a Context,
    hugr: &'a V,
    block: &'b Block<'a>,
    scope: Scope<'a, 'b>,
    symbols: Symboliser<'a, V>,
}

impl<'a, 'b, V: HugrView> Clone for TranslationState<'a, 'b, V> {
    fn clone(&self) -> Self {
        Self {
            context: self.context,
            hugr: self.hugr,
            block: self.block,
            scope: self.scope.clone(),
            symbols: self.symbols.clone(),
            // seen_nodes: self.seen_nodes.clone(),
        }
    }
}

impl<'a, 'b, V: HugrView> TranslationState<'a, 'b, V>
where
    'a: 'b,
{
    fn new(context: &'a Context, hugr: &'a V, block: &'b Block<'a>) -> Self {
        TranslationState {
            context,
            hugr,
            block,
            scope: HashMap::new(),
            symbols: Symboliser::new(context, hugr),
            // scope: Cow::Owned(HashMap::new()),
            // symbols: Cow::Owned(HashMap::new()),
            // seen_nodes: Cow::Owned(std::collections::HashSet::new()),
        }
    }

    pub fn push_scope(&mut self, binders: impl Iterator<Item = ScopeItem<'a, 'b>>) {
        for (k, v) in binders {
            // self.scope.to_mut().insert(k, v);
            self.scope.insert(k, v);
        }
    }

    fn push_block<'c, F, T>(
        &self,
        block: &'c Block<'a>,
        block_args: impl IntoIterator<Item = (hugr::Node, hugr::Port)>,
        f: F,
    ) -> T
    where
        'b: 'c,
        // T: 'c + 'a,
        F: FnOnce(TranslationState<'a, 'c, V>) -> T,
    {
        let mut state = TranslationState {
            context: self.context,
            hugr: self.hugr,
            block,
            scope: self.scope.clone(),
            symbols: self.symbols.clone(),
            // seen_nodes: self.seen_nodes,
        };
        state.push_scope(
            block_args
                .into_iter()
                .enumerate()
                .map(|(i, k)| (k, block.argument(i).unwrap().into())),
        );
        f(state)
    }

    fn lookup_nodeport(&'_ self, n: hugr::Node, p: hugr::Port) -> Value<'a, 'b> {
        *self
            .scope
            .get(&(n, p))
            .unwrap_or_else(|| panic!("lookup_nodeport: {:?}\n{:?}", (n, p), &self.scope))
    }

    fn push_operation(
        &mut self,
        n: hugr::Node,
        result_ports: impl IntoIterator<Item = hugr::Port>,
        op: impl Into<Operation<'a>>,
    ) -> Result<()> {
        let op_ref = self.block.append_operation(op.into());
        let u = unsafe { op_ref.to_ref() };
        let outputs = zip_eq(result_ports.into_iter(), u.results())
            .map(|(p, v)| ((n, p), Into::<Value<'a, 'b>>::into(v)))
            .collect_vec();
        self.push_scope(outputs.into_iter());
        Ok(())
    }




    fn collect_inputs_vec(
        &self,
        n: hugr::Node,
    ) -> Result<Vec<(hugr::Port, melior::ir::Value<'a, 'b>)>> {
        self.collect_inputs::<Vec<_>>(n)
    }

    fn collect_inputs<R: FromIterator<(hugr::Port, melior::ir::Value<'a, 'b>)>>(
        &self,
        n: hugr::Node,
    ) -> Result<R> {
        let optype = self.hugr.get_optype(n);
        self.hugr
            .node_inputs(n)
            .filter_map(|p| match optype.port_kind(p) {
                Some(EdgeKind::Value { .. }) => Some(
                    match self
                        .hugr
                        .linked_ports(n, p)
                        .collect::<std::vec::Vec<_>>()
                        .as_slice()
                    {
                        [(m, x)] => Ok((p, self.lookup_nodeport(*m, *x))),
                        _ => Err(anyhow!("Not a unique link to input port")),
                    },
                ),
                _ => None,
            })
            .collect()
    }

    fn collect_outputs_vec(&self, n: hugr::Node) -> Result<Vec<(hugr::Port, Type<'a>)>> {
        self.collect_outputs::<Vec<_>>(n)
    }

    fn collect_outputs<R: FromIterator<(hugr::Port, Type<'a>)>>(&self, n: hugr::Node) -> Result<R> {
        let optype = self.hugr.get_optype(n);
        self.hugr
            .node_outputs(n)
            .filter_map(|p| match optype.port_kind(p) {
                Some(EdgeKind::Value(ref ty)) => {
                    Some(hugr_to_mlir_type(self.context, ty).map(|x| (p, x)))
                }
                _ => None,
            })
            .collect::<Result<R>>()
    }

    fn node_to_op(&mut self, n: hugr::Node, loc: Location<'a>) -> Result<()> {
        use hugr::ops::OpType;
        let (input_ports, input_values): (Vec<_>,Vec<_>) = self.collect_inputs_vec(n)?.into_iter().unzip();
        let (output_ports, output_types): (Vec<_>,Vec<_>) =  self.collect_outputs_vec(n)?.into_iter().unzip();
        let op: melior::ir::Operation<'_> = self.hugr.get_optype(n).emit(self, n, output_types.as_slice(), input_values.as_slice(), loc)?.into();
        self.push_operation(n, output_ports, op)
    }

    fn build_dataflow_block<
        'c,
    >(&self,
        parent: hugr::Node,
        block: &Block<'a>,
    ) -> Result<()> {
        let ul = melior::ir::Location::unknown(self.context);
        self.build_dataflow_block_term(parent, block, |mut state, vs,n| {
                let op = self.hugr.get_optype(n).emit(&mut state, n, &[], vs, ul)?;
                Ok(((), op.into()))
        })
    }

    fn build_dataflow_block_term<
        'c,
        T,
        F: FnOnce(TranslationState<'a, 'c, V>, &[Value<'a, 'c>], hugr::Node) -> Result<(T, Operation<'c>)>,
    >(
        &self,
        parent: hugr::Node,
        block: &'c Block<'a>,
        mk_terminator: F,
    ) -> Result<T>
    where
        'b: 'c,
    {
        assert!(block.argument_count() == 0);
        let ul = melior::ir::Location::unknown(self.context);
        let [i, o] = self
            .hugr
            .get_io(parent)
            .ok_or(anyhow!("FuncDefn has no io nodes"))?;

        let input_type = self.hugr.get_optype(i);
        let block_arg_port_type_loc = self
            .hugr
            .node_outputs(i)
            .filter_map(|p| match input_type.port_kind(p) {
                Some(hugr::types::EdgeKind::Value(t)) => Some((p, t)),
                _ => None,
            })
            .map(|(p, ref t)| Ok((p, hugr_to_mlir_type(self.context, t)?, ul)))
            .collect::<Result<Vec<(hugr::Port,melior::ir::Type<'a>, melior::ir::Location<'a>)>,Error>>()?;
        for (_, t, l) in block_arg_port_type_loc.iter() {
            block.add_argument(*t, *l);
        }
        let t = {
            let it = block_arg_port_type_loc
                .into_iter()
                .map(|(p, _, _)| (i, p))
                .collect_vec();

            self.clone().push_block(block, it, |mut state| {
                for c in state.hugr.children(parent).filter(|x| *x != i && *x != o) {
                    state.node_to_op(c, ul)?;
                }
                let inputs = state
                    .collect_inputs_vec(o)?
                    .into_iter()
                    .map(|(_, v)| v)
                    .collect_vec();
                let (t, op) = mk_terminator(state, inputs.as_slice(), o)?;
                block.append_operation(op);
                Ok::<_, Error>(t)
            })?
        };
        Ok(t)
    }
}

/// creates an MLIR module holding the mlir representation of the HugrView
pub fn hugr_to_mlir<'c>(
    loc: Location<'c>,
    hugr: &impl HugrView,
) -> Result<melior::ir::Module<'c>, Error> {
    let module = melior::ir::Module::new(loc);
    let block = module.body();
    let context = loc.context();
    let mut state = TranslationState::new(&context, hugr, &block);
    state.node_to_op(hugr.root(), loc)?;
    Ok(module)
}

#[cfg(test)]
mod test {
    use crate::{Result, mlir};
    use crate::{mlir::test::test_context, test::example_hugrs};
    use hugr::extension::ExtensionRegistry;
    use rstest::{fixture, rstest};

    #[rstest]
    fn test_simple_recursion(test_context: melior::Context) -> Result<()> {
        let hugr = example_hugrs::simple_recursion()?;
        let ul = melior::ir::Location::unknown(&test_context);
        let mut op = super::hugr_to_mlir(ul, &hugr)?;
        assert!(mlir::hugr_passes::verify_op(&mut op).is_ok());
        insta::assert_snapshot!(op.as_operation().to_string());
        Ok(())
    }

    #[rstest]
    fn test_cfg(test_context: melior::Context) -> Result<()> {
        let hugr = example_hugrs::cfg()?;
        let ul = melior::ir::Location::unknown(&test_context);
        let mut op = super::hugr_to_mlir(ul, &hugr)?;
        assert!(mlir::hugr_passes::verify_op(&mut op).is_ok());
        insta::assert_snapshot!(op.as_operation().to_string());
        Ok(())
    }

    // This fails because the root doesn't have the right number of ports
    // #[rstest]
    // fn test_basic_loop(test_context: melior::Context) -> Result<()> {
    //     let hugr = example_hugrs::basic_loop()?;
    //     let er = &hugr::extension::prelude::PRELUDE_REGISTRY;
    //     hugr.validate(er)?;
    //     let ul = melior::ir::Location::unknown(&test_context);
    //   let op = super::hugr_to_mlir(ul, &hugr)?;
    //     assert!(op.as_operation().verify());
    //     insta::assert_snapshot!(op.as_operation().to_string());
    //     Ok(())
    // }

    #[rstest]
    fn test_loop_with_conditional(test_context: melior::Context) -> Result<()> {
        let hugr = example_hugrs::loop_with_conditional()?;
        let ul = melior::ir::Location::unknown(&test_context);
        let mut op = super::hugr_to_mlir(ul, &hugr)?;
        assert!(mlir::hugr_passes::verify_op(&mut op).is_ok());
        insta::assert_snapshot!(op.as_operation().to_string());
        Ok(())
    }
}
