use anyhow::anyhow;
use hugr::ops::{OpTrait, OpType};
use itertools::{zip_eq, Itertools};
use melior::ir::attribute::FlatSymbolRefAttribute;
use melior::ir::{BlockRef, Type, ValueLike};
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

use crate::mlir::hugr::{StaticEdgeAttr, DfgOp};
use crate::{mlir, Error, Result};

pub mod types;
use types::*;

pub mod value;
use value::*;

use hugr::ops::dataflow::DataflowOpTrait;

type Scope<'a, 'b> = HashMap<(hugr::Node, hugr::OutgoingPort), Value<'a, 'b>>;
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
                }) if signature.params().len() == 0 => {
                    let ty = hugr_to_mlir_function_type(self.context, signature.body())?.into();
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

struct MlirData<'a, 'b> {
    node: hugr::Node,
    result_types: Vec<Type<'a>>,
    inputs: Vec<Value<'a, 'b>>,
    loc: Location<'a>,
}

trait EmitMlir {
    type Op<'a>: Into<Operation<'a>>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>>;
}

impl EmitMlir for hugr::ops::OpType {
    type Op<'a> = Operation<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::OpType;
        match self {
            OpType::Conditional(ref conditional) => conditional.emit(state, data).map(Into::into),
            OpType::Output(ref output) => output.emit(state, data).map(Into::into),
            OpType::FuncDecl(ref funcdecl) => funcdecl.emit(state, data).map(Into::into),
            OpType::FuncDefn(ref funcdefn) => funcdefn.emit(state, data).map(Into::into),
            OpType::Module(ref module) => module.emit(state, data).map(Into::into),
            OpType::Call(ref call) => call.emit(state, data).map(Into::into),
            OpType::CallIndirect(ref callindirect) => {
                callindirect.emit(state, data).map(Into::into)
            }
            OpType::CFG(ref cfg) => cfg.emit(state, data).map(Into::into),
            OpType::LeafOp(ref leaf) => leaf.emit(state, data).map(Into::into),
            OpType::Const(ref const_) => const_.emit(state, data).map(Into::into),
            OpType::LoadConstant(ref lc) => lc.emit(state, data).map(Into::into),
            OpType::TailLoop(ref tailloop) => tailloop.emit(state, data).map(Into::into),
            OpType::DFG(ref dfg) => dfg.emit(state, data).map(Into::into),
            _ => todo!(),
        }
    }
}

impl EmitMlir for hugr::ops::DFG {
    type Op<'a> = mlir::hugr::DfgOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        let block = Block::new(&[]);
        state.build_dataflow_block(data.node, &block)?;
        let body = Region::new();
        body.append_block(block);
        let context = unsafe { data.loc.context().to_ref() };
        Ok(mlir::hugr::DfgOp::new(
            body,
            &data.result_types,
            &data.inputs,
            extension_set_to_extension_set_attr(context, &self.signature.extension_reqs),
            data.loc,
        ))
    }
}

impl EmitMlir for hugr::ops::Conditional {
    type Op<'a> = mlir::hugr::ConditionalOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        let cases = state
            .hugr
            .children(data.node)
            .map(|case_n| {
                let r: Region<'a> = Region::new();
                let b = Block::new(&[]);
                state.build_dataflow_block(case_n, &b)?;
                r.append_block(b);
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(mlir::hugr::ConditionalOp::new(
            &data.result_types,
            &data.inputs,
            cases,
            data.loc,
        ))
    }
}

impl EmitMlir for hugr::ops::Output {
    type Op<'a> = mlir::hugr::OutputOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        _state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        Ok(mlir::hugr::OutputOp::new(&data.inputs, data.loc))
    }
}

impl EmitMlir for hugr::ops::FuncDefn {
    type Op<'a> = mlir::hugr::FuncOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        let (_, ty, sym) = state.symbols.get_or_alloc(data.node)?;
        let block = Block::new(&[]);
        state.build_dataflow_block(data.node, &block)?;
        let body = Region::new();
        body.append_block(block);
        Ok(mlir::hugr::FuncOp::new(body, sym, ty.try_into()?, data.loc))
    }
}

impl EmitMlir for hugr::ops::FuncDecl {
    type Op<'a> = mlir::hugr::FuncOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        let (_, ty, sym) = state.symbols.get_or_alloc(data.node)?;
        let body = Region::new();
        Ok(mlir::hugr::FuncOp::new(body, sym, ty.try_into()?, data.loc))
    }
}

impl EmitMlir for hugr::ops::Module {
    type Op<'a> = mlir::hugr::ModuleOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        let body = Region::new();
        let block = Block::new(&[]);
        state.push_block(&block, empty(), |mut state| {
            for c in state.hugr.children(data.node) {
                state.node_to_op(c, data.loc)?;
            }
            Ok::<_, crate::Error>(())
        })?;
        body.append_block(block);
        Ok(mlir::hugr::ModuleOp::new_with_body(body, data.loc))
    }
}

impl EmitMlir for hugr::ops::Call {
    type Op<'a> = mlir::hugr::CallOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::OpType;
        use hugr::PortIndex;
        let static_index = self.signature.input.len();
        let static_port = state
            .hugr
            .node_inputs(data.node)
            .find(|p| p.index() == static_index)
            .ok_or(anyhow!("Failed to find static edge to function"))?;
        assert_eq!(
            state.hugr.get_optype(data.node).port_kind(static_port),
            Some(EdgeKind::Static(hugr::types::Type::new_function(
                self.signature.clone()
            )))
        );
        let (target_n, _) = state
            .hugr
            .linked_ports(data.node, static_port)
            .collect_vec()
            .into_iter()
            .exactly_one()?;
        let (static_edge, _, _) = state.symbols.get_or_alloc(target_n)?;

        Ok(mlir::hugr::CallOp::new(
            static_edge,
            &data.inputs,
            &data.result_types,
            data.loc,
        ))
    }
}

impl EmitMlir for hugr::ops::CallIndirect {
    type Op<'a> = mlir::hugr::CallOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        _state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        Ok(mlir::hugr::CallOp::new_indirect(
            data.inputs[0],
            &data.inputs[1..],
            &data.result_types,
            data.loc,
        ))
    }
}

impl EmitMlir for hugr::ops::CFG {
    type Op<'a> = mlir::hugr::CfgOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::controlflow::BasicBlock;
        use hugr::ops::OpType;
        use hugr::types::EdgeKind;
        use hugr::PortIndex;
        let body: Region<'a> = Region::new();
        {
            let mut node_to_block = HashMap::<hugr::Node, BlockRef<'a, '_>>::new();
            for c in state.hugr.children(data.node) {
                let b = body.append_block(melior::ir::Block::new(&[]));
                node_to_block.insert(c, b);
            }
            for (dfb_node, block) in node_to_block.iter() {
                match state.hugr.get_optype(*dfb_node) {
                    optype @ &OpType::BasicBlock(BasicBlock::DFB { .. }) => state
                        .build_dataflow_block_term(
                            *dfb_node,
                            block.deref(),
                            |state, inputs, _| {
                                let successors = state
                                    .hugr
                                    .node_outputs(*dfb_node)
                                    .filter(|p| {
                                        optype.port_kind(*p)
                                            == Some(hugr::types::EdgeKind::ControlFlow)
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
                                    mlir::hugr::SwitchOp::new(&inputs, &successors, data.loc)
                                        .into(),
                                ))
                            },
                        )?,
                    &OpType::BasicBlock(BasicBlock::Exit { ref cfg_outputs }) => {
                        let args = collect_type_row_vec(state.context, cfg_outputs)?
                            .into_iter()
                            .map(|t| block.add_argument(t, data.loc))
                            .collect_vec();
                        block.append_operation(mlir::hugr::OutputOp::new(&args, data.loc).into());
                    }
                    &_ => Err(anyhow!("not a basic block"))?,
                };
            }
        }
        Ok(mlir::hugr::CfgOp::new(
            body,
            &data.result_types,
            &data.inputs,
            data.loc,
        ))
    }
}

impl EmitMlir for hugr::ops::custom::ExternalOp {
    type Op<'a> = mlir::hugr::ExtensionOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::custom::ExternalOp;
        let name = match self {
            ExternalOp::Extension(ref e) => e.def().name(),
            ExternalOp::Opaque(ref o) => o.name(),
        };
        let extensions =
            extension_set_to_extension_set_attr(state.context, &self.signature().extension_reqs);
        Ok(mlir::hugr::ExtensionOp::new(
            &data.result_types,
            name,
            extensions,
            &data.inputs,
            data.loc,
        ))
    }
}

impl EmitMlir for hugr::ops::LeafOp {
    type Op<'a> = melior::ir::Operation<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        use hugr::ops::LeafOp;
        match self {
            LeafOp::CustomOp(custom) => custom.emit(state, data).map(Into::into),
            LeafOp::MakeTuple { .. } => {
                assert_eq!(data.result_types.len(), 1);
                Ok(
                    mlir::hugr::MakeTupleOp::new(data.result_types[0], &data.inputs, data.loc)
                        .into(),
                )
            }
            LeafOp::UnpackTuple { .. } => {
                assert_eq!(data.inputs.len(), 1);
                Ok(
                    mlir::hugr::UnpackTupleOp::new(&data.result_types, data.inputs[0], data.loc)
                        .into(),
                )
            }
            LeafOp::Tag { tag, .. } => {
                assert_eq!(data.result_types.len(), 1);
                Ok(mlir::hugr::TagOp::new(
                    data.result_types[0],
                    *tag as u32,
                    &data.inputs,
                    data.loc,
                )
                .into())
            }
            LeafOp::Lift { new_extension, .. } => Ok(mlir::hugr::LiftOp::new(
                &data.result_types,
                &data.inputs,
                extension_id_to_extension_attr(state.context, new_extension),
                data.loc,
            )
            .into()),
            &_ => panic!("Unimplemented leafop: {:?}", self),
        }
    }
}

impl EmitMlir for hugr::ops::Const {
    type Op<'a> = mlir::hugr::ConstOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        let (_, _, name) = state.symbols.get_or_alloc(data.node)?;
        let ty = hugr_to_mlir_type(state.context, self.const_type())?;
        let val = hugr_to_mlir_value(state.context, self.const_type(), self.value())?;
        Ok(mlir::hugr::ConstOp::new(name, ty, val, data.loc))
    }
}

impl EmitMlir for hugr::ops::LoadConstant {
    type Op<'a> = mlir::hugr::LoadConstantOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        use hugr::PortIndex;
        let static_index = 0usize;
        let static_port = state
            .hugr
            .node_inputs(data.node)
            .find(|p| p.index() == static_index)
            .ok_or(anyhow!("Failed to find static edge to function"))?;
        let (target_n, _) = state
            .hugr
            .linked_ports(data.node, static_port)
            .collect_vec()
            .into_iter()
            .exactly_one()?;
        let (edge, _, _) = state.symbols.get_or_alloc(target_n)?;
        assert_eq!(data.result_types.len(), 1);
        Ok(mlir::hugr::LoadConstantOp::new(
            data.result_types[0],
            edge,
            data.loc,
        ))
    }
}

impl EmitMlir for hugr::ops::TailLoop {
    type Op<'a> = mlir::hugr::TailLoopOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        state: &mut TranslationState<'a, 'b, V>,
        data: MlirData<'a, 'b>,
    ) -> Result<Self::Op<'a>> {
        assert_eq!(
            data.result_types.len(),
            self.just_outputs.len() + self.rest.len()
        );
        let passthrough_inputs = &data.inputs[self.just_inputs.len()..];
        let outputs_types = &data.result_types[0..self.just_outputs.len()];
        let block = Block::new(&[]);
        state.build_dataflow_block_term(data.node, &block, |mut state, inputs, out_n| {
            state
                .hugr
                .get_optype(out_n)
                .emit(
                    &mut state,
                    MlirData {
                        node: out_n,
                        result_types: vec![],
                        inputs,
                        loc: data.loc,
                    },
                )
                .map(|x| ((), x))
        })?;
        let body = Region::new();
        body.append_block(block);
        Ok(mlir::hugr::TailLoopOp::new(
            outputs_types,
            &data.inputs,
            passthrough_inputs,
            body,
            data.loc,
        ))
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
        block_args: impl IntoIterator<Item = (hugr::Node, hugr::OutgoingPort)>,
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

    fn lookup_nodeport(&'_ self, n: hugr::Node, p: hugr::OutgoingPort) -> Value<'a, 'b> {
        *self
            .scope
            .get(&(n, p))
            .unwrap_or_else(|| panic!("lookup_nodeport: {:?}\n{:?}", (n, p), &self.scope))
    }

    fn push_operation(
        &mut self,
        n: hugr::Node,
        result_ports: impl IntoIterator<Item = hugr::OutgoingPort>,
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
    ) -> Result<Vec<(hugr::IncomingPort, melior::ir::Value<'a, 'b>)>> {
        self.collect_inputs::<Vec<_>>(n)
    }

    fn collect_inputs<R: FromIterator<(hugr::IncomingPort, melior::ir::Value<'a, 'b>)>>(
        &self,
        n: hugr::Node,
    ) -> Result<R> {
        let optype = self.hugr.get_optype(n);
        self.hugr
            .node_inputs(n)
            .filter_map(|in_p| match optype.port_kind(in_p) {
                Some(EdgeKind::Value { .. }) => self
                    .hugr
                    .single_linked_output(n, in_p)
                    .map(|(m, out_p)| Ok((in_p, self.lookup_nodeport(m, out_p)))),
                _ => None,
            })
            .collect()
    }

    fn collect_outputs_vec(&self, n: hugr::Node) -> Result<Vec<(hugr::OutgoingPort, Type<'a>)>> {
        self.collect_outputs::<Vec<_>>(n)
    }

    fn collect_outputs<R: FromIterator<(hugr::OutgoingPort, Type<'a>)>>(
        &self,
        n: hugr::Node,
    ) -> Result<R> {
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

    fn node_to_op(&mut self, node: hugr::Node, loc: Location<'a>) -> Result<()> {
        use hugr::ops::OpType;
        let (_, inputs): (Vec<_>, Vec<_>) = self.collect_inputs_vec(node)?.into_iter().unzip();
        let (output_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(node)?.into_iter().unzip();
        let op: melior::ir::Operation<'_> = self.hugr.get_optype(node).emit(
            self,
            MlirData {
                node,
                result_types,
                inputs,
                loc,
            },
        )?;
        self.push_operation(node, output_ports, op)
    }

    fn build_dataflow_block(&self, parent: hugr::Node, block: &Block<'a>) -> Result<()> {
        let ul = melior::ir::Location::unknown(self.context);
        self.build_dataflow_block_term(parent, block, |mut state, inputs, node| {
            let op = self.hugr.get_optype(node).emit(
                &mut state,
                MlirData {
                    node,
                    result_types: vec![],
                    inputs: inputs.into_iter().collect_vec(),
                    loc: ul,
                },
            )?;
            Ok(((), op))
        })
    }

    fn build_dataflow_block_term<
        'c,
        T,
        F: FnOnce(
            TranslationState<'a, 'c, V>,
            Vec<Value<'a, 'c>>,
            hugr::Node,
        ) -> Result<(T, Operation<'c>)>,
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
            .collect::<Result<
                Vec<(
                    hugr::OutgoingPort,
                    melior::ir::Type<'a>,
                    melior::ir::Location<'a>,
                )>,
                Error,
            >>()?;

        let dfg_block = Block::new(&[]);
        for (_, t, l) in block_arg_port_type_loc.iter() {
            block.add_argument(*t, *l);
            dfg_block.add_argument(*t, *l);
        }
        {
            let it = block_arg_port_type_loc
                .into_iter()
                .map(|(p, _, _)| (i, p))
                .collect_vec();
            let it_ref = &it;

            self.push_block(block, it_ref.iter().copied(), move |state| {
                // let scoped_defs = {
                //     let defs = self.hugr.children(parent).filter(|x| match self.hugr.get_optype(*x) {
                //         OpType::FuncDefn(_) => true,
                //         OpType::FuncDecl(_) => true,
                //         OpType::AliasDefn(_) => true,
                //         OpType::AliasDecl(_) => true,
                //         _ => false
                //     }).collect::<HashSet<_>>();
                //     for c in &defs {
                //         state.node_to_op(*c, ul)?;
                //     }
                //     defs
                // };
                // let scoped_defs_ref = &scoped_defs;


                // let dfg_body = Region::new();
                // let dfg_br = dfg_body.append_block(dfg_block);
                let output_tys = state.push_block(&dfg_block, it_ref.iter().copied(), move |mut state| {
                    for c in state.hugr.children(parent).filter(|x| *x != i && *x != o) {
                        state.node_to_op(c, ul)?;
                    }
                    let output_args = state
                        .collect_inputs_vec(o)?
                        .into_iter()
                        .map(|(_, v)| v)
                        .collect_vec();
                    state.block.append_operation(mlir::hugr::OutputOp::new(&output_args, ul).into());
                    Ok::<_, Error>(output_args.iter().map(|x| x.r#type()).collect_vec())
                })?;
                let mut dfg_args = Vec::new();
                for i in 0..state.block.argument_count() { dfg_args.push(state.block.argument(i).unwrap().into()) }

                let dfg_body = Region::new();
                dfg_body.append_block(dfg_block);
                let dfg_op = block.append_operation(mlir::hugr::DfgOp::new(dfg_body, &output_tys, &dfg_args, mlir::hugr::ExtensionSetAttr::new(state.context, []), ul).into());
                let mut dfg_results = Vec::<Value<'a,'c>>::new();
                for r in unsafe {dfg_op.to_ref()}.results() {
                    dfg_results.push(r.into());
                }

                let (t, op) = mk_terminator(state, dfg_results, o)?;
                block.append_operation(op);
                Ok(t)
            })
        }
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
    use crate::{mlir, Result};
    use crate::{mlir::test::test_context, test::example_hugrs};
    use hugr::extension::ExtensionRegistry;
    use rstest::{fixture, rstest};

    // TODO test DFG

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

    #[test]
    fn test_loop_with_conditional() -> Result<()> {
        // let dr = melior::dialect::DialectRegistry::new();
        let hugr_dh = crate::mlir::hugr::get_hugr_dialect_handle();
        // hugr_dh.insert_dialect(&dr);
        let ctx = melior::Context::new();
        println!("0: {}", ctx.registered_dialect_count());
        println!("1: {}", ctx.loaded_dialect_count());
        hugr_dh.load_dialect(&ctx);
        // ctx.append_dialect_registry(&dr);
        println!("2: {}", ctx.registered_dialect_count());
        println!("3: {}", ctx.loaded_dialect_count());
        ctx.get_or_load_dialect("hugr");
        println!("4: {}", ctx.registered_dialect_count());
        println!("5: {}", ctx.loaded_dialect_count());

        let hugr = example_hugrs::loop_with_conditional()?;
        println!("dougrulz0");
        let ul = melior::ir::Location::unknown(&ctx);
        let i = melior::ir::Identifier::new(&ctx, "d");
        println!("dougrulz1");

        let mut op = super::hugr_to_mlir(ul, &hugr)?;
        println!("dougrulz2");
        assert!(mlir::hugr_passes::verify_op(&mut op).is_ok());
        insta::assert_snapshot!(op.as_operation().to_string());
        Ok(())
    }
}
