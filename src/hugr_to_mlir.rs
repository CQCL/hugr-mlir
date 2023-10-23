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

    // fn get(&self, k: &hugr::Node) -> Option<&SymbolItem<'a>> {
    //     self.node_to_symbol.get(k)
    // }

    fn get_or_alloc<'b>(&'b mut self, k: hugr::Node) -> Result<SymbolItem<'a>> {
        if let Some(r) = self.node_to_symbol.get(&k) {
            Ok(*r)
        } else {
            use hugr::hugr::NodeIndex;
            use hugr::ops::OpType;
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
    type Op<'a> : Into<Operation<'a>>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        node: hugr::Node,
        state: &mut TranslationState<'a, 'b, V>,
        result_types: &[Type<'a>],
        inputs: &[Value<'a, 'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>>;
}

impl EmitMlir for hugr::ops::Conditional {
    type Op<'a> = mlir::hugr::ConditionalOp<'a>;
    fn emit<'a, 'b, V: HugrView>(
        &self,
        node: hugr::Node,
        state: &mut TranslationState<'a, 'b, V>,
        result_types: &[Type<'a>],
        inputs: &[Value<'a,'b>],
        loc: Location<'a>,
    ) -> Result<Self::Op<'a>> {
        let cases = state
            .hugr
            .children(node)
            .map(|case_n| {
                let r: Region<'a> = Region::new();
                let b = r.append_block(Block::new(&[]));
                state.build_dataflow_block(case_n, b, |outputs, _| {
                    mk_output(outputs, loc).map(|x| ((), x))
                })?;
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(mlir::hugr::ConditionalOp::new(result_types, inputs, cases, loc))
    }
}

struct TranslationState<'a, 'b, V: HugrView>
where
    'a: 'b,
{
    context: &'a Context,
    hugr: &'a V,
    block: BlockRef<'a, 'b>,
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

fn mk_output<'c>(
    outputs: &[melior::ir::Value<'c, '_>],
    loc: melior::ir::Location<'c>,
) -> Result<melior::ir::Operation<'c>, Error> {
    Ok(mlir::hugr::OutputOp::new(outputs, loc).into())
}

impl<'a, 'b, V: HugrView> TranslationState<'a, 'b, V>
where
    'a: 'b,
{
    fn new(context: &'a Context, hugr: &'a V, block: BlockRef<'a, 'b>) -> Self {
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
        self,
        block: BlockRef<'a, 'c>,
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
            scope: self.scope,
            symbols: self.symbols,
            // seen_nodes: self.seen_nodes,
        };
        let b = unsafe { block.to_ref() };
        state.push_scope(
            block_args
                .into_iter()
                .enumerate()
                .map(|(i, k)| (k, b.argument(i).unwrap().into())),
        );
        f(state)
    }

    fn lookup_nodeport(&'_ self, n: hugr::Node, p: hugr::Port) -> Value<'a, 'b> {
        *self
            .scope
            .get(&(n, p))
            .unwrap_or_else(|| panic!("lookup_nodeport: {:?}\n{:?}", (n, p), &self.scope))
    }

    // fn push_operation(
    //     &mut self,
    //     n: hugr::Node,
    //     result_ports: impl Iterator<Item = hugr::Port>,
    //     op: impl Into<Operation<'a>>,
    // ) -> Result<()> {
    //     let _ = self.push_operation2(n, result_ports, op)?;
    //     Ok(())
    // }

    fn push_operation(
        &mut self,
        n: hugr::Node,
        result_ports: impl IntoIterator<Item = hugr::Port>,
        op: impl Into<Operation<'a>>,
    ) -> Result<()> {
        let op_ref = unsafe { self.block.to_ref() }.append_operation(op.into());
        let u = unsafe { op_ref.to_ref() };
        let outputs = zip_eq(result_ports.into_iter(), u.results())
            .map(|(p, v)| ((n, p), Into::<Value<'a, 'b>>::into(v)))
            .collect_vec();
        self.push_scope(outputs.into_iter());
        Ok(())
    }

    fn mk_function_defn(&mut self, n: hugr::Node, loc: melior::ir::Location<'a>) -> Result<()> {
        let (_, ty, sym) = self.symbols.get_or_alloc(n)?;
        let body = Region::new();
        let block = body.append_block(Block::new(&[]));
        self.clone()
            .build_dataflow_block(n, block, |inputs, _node| Ok(((), mk_output(inputs, loc)?)))?;
        self.push_operation(
            n,
            empty(),
            mlir::hugr::FuncOp::new(body, sym, ty.try_into()?, loc),
        )
    }

    fn mk_function_decl(&mut self, n: hugr::Node, loc: melior::ir::Location<'a>) -> Result<()> {
        let (_, ty, sym) = self.symbols.get_or_alloc(n)?;
        let body = melior::ir::Region::new();

        self.push_operation(
            n,
            empty(),
            mlir::hugr::FuncOp::new(body, sym, ty.try_into()?, loc),
        )
    }

    fn mk_module(&mut self, n: hugr::Node, loc: melior::ir::Location<'a>) -> Result<()> {
        let body = Region::new();
        let block = body.append_block(Block::new(&[]));
        {
            // let mut state: TranslationState<'a, '_, V> = self.clone();
            // state = state.push_block(&block, empty());
            // for c in state.hugr.children(n) {
            //     state = state.node_to_op(c, loc)?;
            // }
            self.clone().push_block(block, empty(), |mut state| {
                for c in state.hugr.children(n) {
                    state.node_to_op(c, loc)?;
                }
                Ok::<_, Error>(())
            })?;
        }
        let op = mlir::hugr::ModuleOp::new_with_body(body, loc);
        self.push_operation(n, empty(), op)
    }

    fn mk_call(
        &mut self,
        call_n: hugr::Node,
        optype: &hugr::ops::OpType,
        loc: melior::ir::Location<'a>,
    ) -> Result<()> {
        use hugr::hugr::PortIndex;
        use hugr::ops::OpType;
        let args_ports = self.collect_inputs_vec(call_n)?;
        let args = args_ports.into_iter().map(|(_, v)| v).collect_vec();
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(call_n)?.into_iter().unzip();
        let op = match *optype {
            OpType::Call(hugr::ops::Call { ref signature }) => {
                let static_index = signature.input.len();
                let static_port = self
                    .hugr
                    .node_inputs(call_n)
                    .find(|p| p.index() == static_index)
                    .ok_or(anyhow!("Failed to find static edge to function"))?;
                assert_eq!(
                    optype.port_kind(static_port),
                    Some(EdgeKind::Static(hugr::types::Type::new_function(
                        signature.clone()
                    )))
                );
                let (target_n, _) = self
                    .hugr
                    .linked_ports(call_n, static_port)
                    .collect_vec()
                    .into_iter()
                    .exactly_one()?;
                let (static_edge, _, _) = self.symbols.get_or_alloc(target_n)?;

                mlir::hugr::CallOp::new(
                    static_edge,
                    args.as_slice(),
                    result_types.as_slice(),
                    loc,
                )
            }
            OpType::CallIndirect { .. } => {
                mlir::hugr::CallOp::new_indirect(args[0], &args[1..], result_types.as_slice(), loc)
            }
            _ => Err(anyhow!("mk_call received bad optype: {}", optype.tag()))?,
        };

        self.push_operation(call_n, result_ports.into_iter(), op)
    }

    // fn new_name(&self, prefix: impl AsRef<str> ) -> String {
    //     let mut x = self.name_supply.borrow_mut();
    //     *x = *x + 1;
    //     format!("{}_{}", prefix.as_ref(), *x)
    // }

    fn mk_cfg(
        &mut self,
        n: hugr::Node,
        _sig: &hugr::types::FunctionType,
        loc: Location<'a>,
    ) -> Result<()> {
        use hugr::hugr::PortIndex;
        use hugr::ops::controlflow::BasicBlock;
        use hugr::ops::OpType;
        use hugr::types::EdgeKind;
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(n)?.into_iter().unzip();
        let inputs = self
            .collect_inputs_vec(n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        let body: Region<'a> = Region::new();
        {
            let mut node_to_block = HashMap::<hugr::Node, BlockRef<'a, '_>>::new();
            for c in self.hugr.children(n) {
                let b = body.append_block(melior::ir::Block::new(&[]));
                node_to_block.insert(c, b);
            }
            for (dfb_node, block) in node_to_block.iter() {
                match self.hugr.get_optype(*dfb_node) {
                    optype @ &OpType::BasicBlock(BasicBlock::DFB { .. }) => self
                        .build_dataflow_block(*dfb_node, *block, |inputs, _| {
                            let successors = self
                                .hugr
                                .node_outputs(*dfb_node)
                                .filter(|p| {
                                    optype.port_kind(*p) == Some(hugr::types::EdgeKind::ControlFlow)
                                })
                                .map(|p| {
                                    Ok(node_to_block[&self
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
                        let args = collect_type_row_vec(self.context, cfg_outputs)?
                            .into_iter()
                            .map(|t| block.add_argument(t, loc))
                            .collect_vec();
                        block.append_operation(mlir::hugr::OutputOp::new(&args, loc).into());
                    }
                    &_ => Err(anyhow!("not a basic block"))?,
                };
            }
        }
        let cfg = mlir::hugr::CfgOp::new(body, result_types.as_slice(), inputs.as_slice(), loc);
        self.push_operation(n, result_ports.into_iter(), cfg)
    }

    fn mk_make_tuple(&mut self, n: hugr::Node, loc: Location<'a>) -> Result<()> {
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(n)?.into_iter().unzip();
        let inputs = self
            .collect_inputs_vec(n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        self.push_operation(
            n,
            result_ports.into_iter(),
            mlir::hugr::MakeTupleOp::new(result_types[0], inputs.as_slice(), loc),
        )
    }

    fn mk_unpack_tuple(&mut self, n: hugr::Node, loc: Location<'a>) -> Result<()> {
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(n)?.into_iter().unzip();
        let inputs = self
            .collect_inputs_vec(n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        self.push_operation(
            n,
            result_ports.into_iter(),
            mlir::hugr::UnpackTupleOp::new(result_types.as_slice(), inputs[0], loc),
        )
    }

    fn mk_const(
        &mut self,
        n: hugr::Node,
        value: &hugr::values::Value,
        typ: &hugr::types::Type,
        loc: Location<'a>,
    ) -> Result<()> {
        let (_, _, name) = self.symbols.get_or_alloc(n)?;
        let ty = hugr_to_mlir_type(self.context, typ)?;
        let val = hugr_to_mlir_value(self.context, typ, value)?;
        self.push_operation(
            n,
            empty(),
            mlir::hugr::ConstOp::new(name, ty, val, loc),
        )
    }

    fn mk_tag(&mut self, n: hugr::Node, tag: usize, loc: Location<'a>) -> Result<()> {
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(n)?.into_iter().unzip();
        let inputs = self
            .collect_inputs_vec(n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        self.push_operation(
            n,
            result_ports.into_iter(),
            mlir::hugr::TagOp::new(result_types[0], tag as u32, inputs.as_slice(), loc),
        )
    }

    fn mk_load_constant(&mut self, lc_n: hugr::Node, loc: Location<'a>) -> Result<()> {
        use hugr::hugr::PortIndex;
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(lc_n)?.into_iter().unzip();
        let static_index = 0usize;
        let static_port = self
            .hugr
            .node_inputs(lc_n)
            .find(|p| p.index() == static_index)
            .ok_or(anyhow!("Failed to find static edge to function"))?;
        let (target_n, _) = self
            .hugr
            .linked_ports(lc_n, static_port)
            .collect_vec()
            .into_iter()
            .exactly_one()?;
        let (edge, _, _) = self.symbols.get_or_alloc(target_n)?;
        self.push_operation(
            lc_n,
            result_ports.into_iter(),
            mlir::hugr::LoadConstantOp::new(result_types[0], edge, loc),
        )
    }

    fn mk_custom_op(
        &mut self,
        co_n: hugr::Node,
        external_op: &hugr::ops::custom::ExternalOp,
        loc: Location<'a>,
    ) -> Result<()> {
        use hugr::hugr::PortIndex;
        let inputs = self
            .collect_inputs_vec(co_n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(co_n)?.into_iter().unzip();
        let extensions = extension_set_to_extension_set_attr(
            self.context,
            &external_op.signature().extension_reqs,
        );
        let name = match external_op {
            hugr::ops::custom::ExternalOp::Extension(ref e) => e.def().name(),
            hugr::ops::custom::ExternalOp::Opaque(ref o) => o.name(),
        };
        self.push_operation(
            co_n,
            result_ports.into_iter(),
            mlir::hugr::ExtensionOp::new(
                result_types.as_slice(),
                name,
                extensions,
                inputs.as_slice(),
                loc,
            ),
        )
    }

    fn mk_conditional(
        &mut self,
        c_n: hugr::Node,
        _conditional: &hugr::ops::controlflow::Conditional,
        loc: Location<'a>,
    ) -> Result<()> {
        let inputs = self
            .collect_inputs_vec(c_n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(c_n)?.into_iter().unzip();
        let cases = self
            .hugr
            .children(c_n)
            .map(|case_n| {
                let r: Region<'a> = Region::new();
                let b = r.append_block(Block::new(&[]));
                self.build_dataflow_block(case_n, b, |outputs, _| {
                    mk_output(outputs, loc).map(|x| ((), x))
                })?;
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?;
        self.push_operation(
            c_n,
            result_ports.into_iter(),
            mlir::hugr::ConditionalOp::new(result_types.as_slice(), inputs.as_slice(), cases, loc),
        )
    }

    fn mk_tail_loop(
        &mut self,
        tl_n: hugr::Node,
        tailloop: &hugr::ops::TailLoop,
        loc: Location<'a>,
    ) -> Result<()> {
        let all_inputs = self
            .collect_inputs_vec(tl_n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(tl_n)?.into_iter().unzip();
        // assert_eq!(
        //     self.hugr.node_outputs(tl_n).collect_vec().len(),
        //     result_types.len()
        // );
        assert_eq!(
            result_types.len(),
            tailloop.just_outputs.len() + tailloop.rest.len()
        );
        let inputs = &all_inputs[0..tailloop.just_inputs.len()];
        let passthrough_inputs = &all_inputs[tailloop.just_inputs.len()..];
        let outputs_types = &result_types[0..tailloop.just_outputs.len()];
        let body = Region::new();
        let block = body.append_block(Block::new(&[]));
        self.build_dataflow_block(tl_n, block, |outputs, _| {
            mk_output(outputs, loc).map(|x| ((), x))
        })?;
        self.push_operation(
            tl_n,
            result_ports,
            mlir::hugr::TailLoopOp::new(outputs_types, inputs, passthrough_inputs, body, loc),
        )
    }

    fn mk_lift(
        &mut self,
        l_n: hugr::Node,
        extension: hugr::extension::ExtensionId,
        loc: Location<'a>,
    ) -> Result<()> {
        let inputs = self
            .collect_inputs_vec(l_n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(l_n)?.into_iter().unzip();
        self.push_operation(
            l_n,
            result_ports,
            mlir::hugr::LiftOp::new(
                result_types.as_slice(),
                inputs.as_slice(),
                extension_id_to_extension_attr(self.context, &extension),
                loc,
            ),
        )
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
        self
            .hugr
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
        let optype = self.hugr.get_optype(n);
        // dbg!(n, optype.tag());
        match optype {
            &OpType::Module(_) => self.mk_module(n, loc),
            &OpType::FuncDefn(_) => self.mk_function_defn(n, loc),
            &OpType::FuncDecl(_) => self.mk_function_decl(n, loc),
            &OpType::Call { .. } | &OpType::CallIndirect { .. } => self.mk_call(n, optype, loc),
            &OpType::CFG(hugr::ops::CFG { ref signature }) => self.mk_cfg(n, signature, loc),
            &OpType::LeafOp(hugr::ops::LeafOp::MakeTuple { .. }) => self.mk_make_tuple(n, loc),
            &OpType::LeafOp(hugr::ops::LeafOp::UnpackTuple { .. }) => self.mk_unpack_tuple(n, loc),
            &OpType::LeafOp(hugr::ops::LeafOp::Tag { tag, .. }) => self.mk_tag(n, tag, loc),
            &OpType::LeafOp(hugr::ops::LeafOp::Lift {
                ref new_extension, ..
            }) => self.mk_lift(n, new_extension.clone(), loc),
            &OpType::LeafOp(hugr::ops::LeafOp::CustomOp(ref external_op)) => {
                self.mk_custom_op(n, external_op, loc)
            }
            OpType::Const(const_) => {
                self.mk_const(n, const_.value(), const_.const_type(), loc)
            }
            OpType::LoadConstant(_const) => self.mk_load_constant(n, loc),
            OpType::Conditional(conditional) => self.mk_conditional(n, conditional, loc),
            OpType::TailLoop(tailloop) => self.mk_tail_loop(n, tailloop, loc),
            t => panic!("unimplemented: {:?}", t),
        }
    }

    fn build_dataflow_block<
        'c,
        T,
        F: FnOnce(&[Value<'a, 'c>], hugr::Node) -> Result<(T, Operation<'c>)>,
    >(
        &self,
        parent: hugr::Node,
        block: BlockRef<'a, 'c>,
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
            // let mut state = self.clone().push_block(block, it);

            // for c in state.hugr.children(parent).filter(|x| *x != i && *x != o) {
            //     state = state.node_to_op(c, ul)?;
            // }

            self.clone().push_block(block, it, |mut state| {
                for c in state.hugr.children(parent).filter(|x| *x != i && *x != o) {
                    state.node_to_op(c, ul)?;
                }
                let inputs = state
                    .collect_inputs_vec(o)?
                    .into_iter()
                    .map(|(_, v)| v)
                    .collect_vec();
                let (t, op) = mk_terminator(inputs.as_slice(), o)?;
                block.append_operation(op);
                Ok::<_, Error>(t)
            })?
        };
        Ok(t)
    }
}

pub fn hugr_to_mlir<'c>(
    loc: Location<'c>,
    hugr: &impl HugrView,
) -> Result<melior::ir::Module<'c>, Error> {
    let module = melior::ir::Module::new(loc);
    let block = module.body();
    let context = loc.context();
    let mut state = TranslationState::new(&context, hugr, block);
    state.node_to_op(hugr.root(), loc)?;
    Ok(module)
}

#[cfg(test)]
mod test {
    use crate::Result;
    use crate::{mlir::test::test_context, test::example_hugrs};
    use hugr::extension::ExtensionRegistry;
    use rstest::{fixture, rstest};

    #[rstest]
    fn test_simple_recursion(test_context: melior::Context) -> Result<()> {
        let hugr = example_hugrs::simple_recursion()?;
        let ul = melior::ir::Location::unknown(&test_context);
        let op = super::hugr_to_mlir(ul, &hugr)?;
        assert!(op.as_operation().verify());
        insta::assert_snapshot!(op.as_operation().to_string());
        Ok(())
    }

    #[rstest]
    fn test_cfg(test_context: melior::Context) -> Result<()> {
        let hugr = example_hugrs::cfg()?;
        let ul = melior::ir::Location::unknown(&test_context);
        let op = super::hugr_to_mlir(ul, &hugr)?;
        assert!(op.as_operation().verify());
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
    //     let op = super::hugr_to_mlir(ul, &hugr)?;
    //     assert!(op.as_operation().verify());
    //     insta::assert_snapshot!(op.as_operation().to_string());
    //     Ok(())
    // }

    #[rstest]
    fn test_loop_with_conditional(test_context: melior::Context) -> Result<()> {
        let hugr = example_hugrs::loop_with_conditional()?;
        let ul = melior::ir::Location::unknown(&test_context);
        let op = super::hugr_to_mlir(ul, &hugr)?;
        assert!(op.as_operation().verify());
        insta::assert_snapshot!(op.as_operation().to_string());
        Ok(())
    }
}
