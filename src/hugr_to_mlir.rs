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
use std::collections::HashMap;
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

struct TranslationState<'a, 'b, V: HugrView>
where
    'a: 'b,
{
    context: &'a Context,
    hugr: &'a V,
    block: BlockRef<'a,'b>,
    scope: Scope<'a, 'b>,
    symbols: HashMap<hugr::Node, SymbolItem<'a>>,
    // seen_nodes: Cow<'b, std::collections::HashSet<hugr::Node>>
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
    fn new(context: &'a Context, hugr: &'a V, block: BlockRef<'a,'b>) -> Self {
        TranslationState {
            context,
            hugr,
            block,
            scope: HashMap::new(),
            symbols: HashMap::new(),
            // scope: Cow::Owned(HashMap::new()),
            // symbols: Cow::Owned(HashMap::new()),
            // seen_nodes: Cow::Owned(std::collections::HashSet::new()),
        }
    }

    pub fn push_scope(&mut self, binders: impl Iterator<Item = ScopeItem<'a, 'b>>) -> () {
        for (k, v) in binders {
            // self.scope.to_mut().insert(k, v);
            self.scope.insert(k, v);
        }
    }

    fn push_block<'c, F, T>(
        self,
        block: BlockRef<'a,'c>,
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
        result_ports: impl Iterator<Item = hugr::Port>,
        op: impl Into<Operation<'a>>,
    ) -> Result<()> {
        let op_ref = unsafe { self.block.to_ref() }.append_operation(op.into());
        let u = unsafe { op_ref.to_ref() };
        let outputs = zip_eq(result_ports, u.results())
            .map(|(p, v)| ((n, p), Into::<Value<'a, 'b>>::into(v)))
            .collect_vec();
        self.push_scope(outputs.into_iter());
        Ok(())
    }

    fn mk_function_defn(
        mut self,
        n: hugr::Node,
        func: &'b hugr::ops::FuncDefn,
        loc: melior::ir::Location<'a>,
    ) -> Result<Self> {
        let hugr::ops::FuncDefn {
            ref name,
            ref signature,
        } = func;
        let name_attr = melior::ir::attribute::StringAttribute::new(&self.context, &name);
        let context: &'a Context = self.context;
        let type_ = hugr_to_mlir_function_type(context, signature)?;
        let body = Region::new();
        let block = body.append_block(Block::new(&[]));
        {
            // self = new_self;

            // let body = unsafe { op_ref.to_ref() }.region(0)?;
            self.clone()
                .build_dataflow_block(n, block, |inputs, _node| {
                    Ok(((), mk_output(inputs, loc)?))
                })?;
            self.push_operation(
                n,
                empty(),
                mlir::hugr::FuncOp::new(body, name_attr, type_, loc),
            )?;
        }
        Ok(self)
    }

    fn mk_function_decl(
        mut self,
        n: hugr::Node,
        func: &hugr::ops::FuncDecl,
        loc: melior::ir::Location<'a>,
    ) -> Result<Self> {
        let hugr::ops::FuncDecl {
            ref name,
            ref signature,
        } = func;
        let name_attr = melior::ir::attribute::StringAttribute::new(&self.context, &name);
        let type_ = hugr_to_mlir_function_type(&self.context, signature)?;
        let body = melior::ir::Region::new();

        self.push_operation(
            n,
            empty(),
            mlir::hugr::FuncOp::new(body, name_attr, type_, loc),
        )?;
        Ok(self)
    }

    fn mk_module(mut self, n: hugr::Node, loc: melior::ir::Location<'a>) -> Result<Self> {
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
                    state = state.node_to_op(c, loc)?;
                }
                Ok::<_, Error>(())
            })?;
        }
        let op = mlir::hugr::ModuleOp::new_with_body(body, loc);
        self.push_operation(n, empty(), op)?;
        Ok(self)
    }

    fn get_static_edge(&mut self, target_n: hugr::Node) -> Result<SymbolItem<'a>> {
        if let Some(r) = self.symbols.get(&target_n) {
            Ok(*r)
        } else {
            use hugr::ops::OpType;
            let (ty, sym) = match self.hugr.get_optype(target_n) {
                &OpType::FuncDecl(hugr::ops::FuncDecl {
                    ref name,
                    ref signature,
                })
                | &OpType::FuncDefn(hugr::ops::FuncDefn {
                    ref name,
                    ref signature,
                }) => {
                    let ty = hugr_to_mlir_function_type(self.context, signature)?.into();
                    Ok((
                        ty,
                        name.to_string(),
                    ))
                }
                &OpType::Const(ref const_) => {
                    let ty = hugr_to_mlir_type(self.context, const_.const_type())?.into();
                    Ok((
                        ty,
                        format!("const_{:?}", target_n),
                    ))
                }
                opty => Err(anyhow!("Bad optype for static edge: {:?}", opty)),
            }?;
            let edge = mlir::hugr::StaticEdgeAttr::new(ty, mlir::hugr::SymbolRefAttr::new(self.context, sym.as_ref(), empty()));
            let item = (edge, ty, melior::ir::attribute::StringAttribute::new(self.context, &sym));
            // self.symbols.to_mut().insert(target_n, item);
            self.symbols.insert(target_n, item);
            Ok(item)
        }
    }

    fn mk_call(
        mut self,
        call_n: hugr::Node,
        call: &hugr::ops::Call,
        loc: melior::ir::Location<'a>,
    ) -> Result<Self> {
        use hugr::hugr::PortIndex;
        use hugr::ops::OpType;
        let args = self.collect_inputs_vec(call_n)?;
        let optype = Into::<OpType>::into(call.clone());
        let static_index = call.signature.input.len();
        let static_port = self
            .hugr
            .node_inputs(call_n)
            .find(|p| p.index() == static_index)
            .ok_or(anyhow!("Failed to find static edge to function"))?;
        assert_eq!(
            optype.port_kind(static_port),
            Some(EdgeKind::Static(hugr::types::Type::new_function(
                call.signature.clone()
            )))
        );
        let (target_n, _) = self
            .hugr
            .linked_ports(call_n, static_port)
            .collect_vec()
            .into_iter()
            .exactly_one()?;
        let (static_edge, _, _) = self.get_static_edge(target_n)?;

        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(call_n)?.into_iter().unzip();
        let call = mlir::hugr::CallOp::new(
            static_edge.into(),
            args.into_iter()
                .map(|(_, v)| v)
                .collect::<Vec<_>>()
                .as_slice(),
            result_types.as_slice(),
            loc,
        );

        self.push_operation(call_n, result_ports.into_iter(), call)?;
        Ok(self)
    }

    // fn new_name(&self, prefix: impl AsRef<str> ) -> String {
    //     let mut x = self.name_supply.borrow_mut();
    //     *x = *x + 1;
    //     format!("{}_{}", prefix.as_ref(), *x)
    // }

    fn mk_cfg(
        mut self,
        n: hugr::Node,
        _sig: &hugr::types::FunctionType,
        loc: Location<'a>,
    ) -> Result<Self> {
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
            let mut node_to_block = HashMap::<hugr::Node, BlockRef<'a,'_>>::new();
            for c in self.hugr.children(n) {
                let b = body.append_block(melior::ir::Block::new(&[]));
                node_to_block.insert(c, b);
            }
            for (dfb_node, block) in node_to_block.iter() {
                self = match self.hugr.get_optype(*dfb_node) {
                    optype @ &OpType::BasicBlock(BasicBlock::DFB { .. }) => {
                        self.clone()
                            .build_dataflow_block(*dfb_node, *block, |inputs, _| {
                                let successors = self
                                    .hugr
                                    .node_outputs(*dfb_node)
                                    .filter(|p| {
                                        optype.port_kind(*p)
                                            == Some(hugr::types::EdgeKind::ControlFlow)
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
                            })?
                            .1
                    }
                    &OpType::BasicBlock(BasicBlock::Exit { ref cfg_outputs }) => {
                        let args = collect_type_row_vec(self.context, cfg_outputs)?
                            .into_iter()
                            .map(|t| block.add_argument(t, loc))
                            .collect_vec();
                        block.append_operation(mlir::hugr::OutputOp::new(&args, loc).into());
                        self
                    }
                    &_ => Err(anyhow!("not a basic block"))?,
                };
            }
        }
        let cfg = mlir::hugr::CfgOp::new(body, result_types.as_slice(), inputs.as_slice(), loc);
        self.push_operation(n, result_ports.into_iter(), cfg)?;
        Ok(self)
    }

    fn mk_make_tuple(mut self, n: hugr::Node, loc: Location<'a>) -> Result<Self> {
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
            mlir::hugr::MakeTupleOp::new(result_types[0].into(), inputs.as_slice(), loc),
        )?;
        Ok(self)
    }

    fn mk_const(
        mut self,
        n: hugr::Node,
        value: &hugr::values::Value,
        typ: &hugr::types::Type,
        loc: Location<'a>,
    ) -> Result<Self> {
        let (_, _, name) = self.get_static_edge(n)?;
        let ty = hugr_to_mlir_type(self.context, typ)?;
        let val = hugr_to_mlir_value(self.context, typ, value)?;
        self.push_operation(n, empty(), mlir::hugr::ConstOp::new(name, ty, val, loc))?;
        Ok(self)
    }

    fn mk_tag(mut self, n: hugr::Node, tag: usize, loc: Location<'a>) -> Result<Self> {
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
        )?;
        Ok(self)
    }

    fn mk_load_constant(mut self, lc_n: hugr::Node, loc: Location<'a>) -> Result<Self> {
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
        let (edge, _, _) = self.get_static_edge(target_n)?;
        self.push_operation(
            lc_n,
            result_ports.into_iter(),
            mlir::hugr::LoadConstantOp::new(result_types[0], edge, loc),
        )?;
        Ok(self)
    }

    fn mk_custom_op(mut self, co_n: hugr::Node, external_op: &hugr::ops::custom::ExternalOp, loc: Location<'a>) -> Result<Self> {
        use hugr::hugr::PortIndex;
        let inputs = self
            .collect_inputs_vec(co_n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(co_n)?.into_iter().unzip();
        let extensions = extension_set_to_extension_set_attr(self.context, &external_op.signature().extension_reqs);
        let name = match external_op {
           hugr::ops::custom::ExternalOp::Extension(ref e)  => e.def().name(),
           hugr::ops::custom::ExternalOp::Opaque(ref o)  => o.name()
        };
        self.push_operation(co_n, result_ports.into_iter(), mlir::hugr::ExtensionOp::new(result_types.as_slice(), name, extensions, inputs.as_slice(), loc))?;
        Ok(self)
    }

    fn mk_conditional(mut self, c_n: hugr::Node, conditional: &hugr::ops::controlflow::Conditional, loc: Location<'a>) -> Result<Self> {
        let optype = hugr::ops::OpType::Conditional(conditional.clone());
        let inputs = self
            .collect_inputs_vec(c_n)?
            .into_iter()
            .map(|(_, v)| v)
            .collect_vec();
        let (result_ports, result_types): (Vec<_>, Vec<_>) =
            self.collect_outputs_vec(c_n)?.into_iter().unzip();
        let self_ref = &mut self;
        let cases = self_ref.hugr.children(c_n).map(|case_n| {
            let r: Region<'a> = Region::new();
            let b = r.append_block(Block::new(&[]));
            let ((), new_self) = self_ref.clone().build_dataflow_block(case_n, b, |outputs,_| mk_output(outputs, loc).map(|x|((),x)))?;
            *self_ref = new_self;
            Ok(r)
        }).collect::<Result<Vec<_>>>()?;
        self.push_operation(c_n, result_ports.into_iter(), mlir::hugr::ConditionalOp::new(result_types.as_slice(), inputs.as_slice(), cases, loc))?;

        Ok(self)
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
        Ok(self
            .hugr
            .node_outputs(n)
            .filter_map(|p| match optype.port_kind(p) {
                Some(EdgeKind::Value(ref ty)) => {
                    Some(hugr_to_mlir_type(self.context, ty).map(|x| (p, x)))
                }
                _ => None,
            })
            .collect::<Result<R>>()?)
    }

    fn node_to_op(mut self, n: hugr::Node, loc: Location<'a>) -> Result<Self, Error> {
        // assert!(!self.seen_nodes.contains(&n));
        // self.seen_nodes.to_mut().insert(n);
        use hugr::ops::OpType;
        let optype = self.hugr.get_optype(n);
        dbg!(n, optype.tag());
        match optype {
            &OpType::Module(_) => self.mk_module(n, loc),
            &OpType::FuncDefn(ref defn) => self.mk_function_defn(n, defn, loc),
            &OpType::FuncDecl(ref decl) => self.mk_function_decl(n, decl, loc),
            &OpType::Call(ref call) => self.mk_call(n, call, loc),
            &OpType::CFG(hugr::ops::CFG { ref signature }) => self.mk_cfg(n, signature, loc),
            &OpType::LeafOp(hugr::ops::LeafOp::MakeTuple { .. }) => self.mk_make_tuple(n, loc),
            &OpType::LeafOp(hugr::ops::LeafOp::Tag { tag, .. }) => self.mk_tag(n, tag, loc),
            &OpType::LeafOp(hugr::ops::LeafOp::CustomOp(ref external_op)) => self.mk_custom_op(n, external_op, loc),
            &OpType::Const(ref const_) => {
                self.mk_const(n, const_.value(), const_.const_type(), loc)
            }
            &OpType::LoadConstant(ref _const) => self.mk_load_constant(n, loc),
            &OpType::Conditional(ref conditional) => self.mk_conditional(n, conditional, loc),
            t => panic!("unimplemented: {:?}", t),
        }
    }

    fn build_dataflow_block<
        'c,
        T,
        F: FnOnce(&[Value<'a, 'c>], hugr::Node) -> Result<(T, Operation<'c>)>,
    >(
        self,
        parent: hugr::Node,
        block: BlockRef<'a,'c>,
        mk_terminator: F,
    ) -> Result<(T, Self)>
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
                    state = state.node_to_op(c, ul)?;
                }
                let inputs = state
                    .collect_inputs_vec(o)?
                    .into_iter()
                    .map(|(_, v)| v)
                    .collect_vec();
                let (t, op) = mk_terminator(inputs.as_slice(), o)?;
                block.append_operation(op);
                Ok::<_,Error>(t)
            })?
        };
        Ok((t, self))
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
    for n in state.hugr.children(hugr.root()) {
        state = state.node_to_op(n, loc)?;
    }
    Ok(module)
}

#[cfg(test)]
mod test {

    use crate::mlir::test::get_test_context;
    use hugr::builder::{
        BuildError, CFGBuilder, Container, Dataflow, DataflowSubContainer, FunctionBuilder,
        HugrBuilder, ModuleBuilder, SubContainer,
    };
    use hugr::extension::{prelude, ExtensionSet};
    use hugr::hugr::ValidationError;
    use hugr::types::{FunctionType, Type};
    use hugr::{type_row, Hugr};
    pub(super) const NAT: hugr::types::Type = hugr::extension::prelude::USIZE_T;

    fn get_example_hugr() -> Result<Hugr, BuildError> {
        const NAT: Type = prelude::USIZE_T;
        let mut module_builder = ModuleBuilder::new();

        let f_id = module_builder.declare(
            "main",
            FunctionType::new(type_row![NAT], type_row![NAT]).pure(),
        )?;

        let mut f_build = module_builder.define_declaration(&f_id)?;
        let call = f_build.call(&f_id, f_build.input_wires())?;

        f_build.finish_with_outputs(call.outputs())?;
        module_builder.finish_prelude_hugr().map_err(|x| x.into())
    }

    fn build_basic_cfg<T: AsMut<Hugr> + AsRef<Hugr>>(
        cfg_builder: &mut CFGBuilder<T>,
    ) -> Result<(), BuildError> {
        let sum2_variants = vec![type_row![NAT], type_row![NAT]];
        let mut entry_b =
            cfg_builder.entry_builder(sum2_variants.clone(), type_row![], ExtensionSet::new())?;
        let entry = {
            let [inw] = entry_b.input_wires_arr();

            let sum = entry_b.make_predicate(1, sum2_variants, [inw])?;
            entry_b.finish_with_outputs(sum, [])?
        };
        let mut middle_b = cfg_builder
            .simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?;
        let middle = {
            let c = middle_b.add_load_const(
                hugr::ops::Const::simple_unary_predicate(),
                ExtensionSet::new(),
            )?;
            let [inw] = middle_b.input_wires_arr();
            middle_b.finish_with_outputs(c, [inw])?
        };
        let exit = cfg_builder.exit_block();
        cfg_builder.branch(&entry, 0, &middle)?;
        cfg_builder.branch(&middle, 0, &exit)?;
        cfg_builder.branch(&entry, 1, &exit)?;
        Ok(())
    }

    fn get_example_hugr_cfg() -> std::result::Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();
        let mut func_builder = module_builder
            .define_function("main", FunctionType::new(vec![NAT], type_row![NAT]).pure())?;
        let _f_id = {
            let [int] = func_builder.input_wires_arr();

            let cfg_id = {
                let mut cfg_builder = func_builder.cfg_builder(
                    vec![(NAT, int)],
                    None,
                    type_row![NAT],
                    ExtensionSet::new(),
                )?;
                build_basic_cfg(&mut cfg_builder)?;

                cfg_builder.finish_sub_container()?
            };

            func_builder.finish_with_outputs(cfg_id.outputs())?
        };
        module_builder.finish_prelude_hugr().map_err(|x| x.into())
    }

    #[test]
    fn test_example_hugr() {
        let ctx = get_test_context();
        let h = get_example_hugr().unwrap();
        let loc = melior::ir::Location::unknown(&ctx);
        let m = melior::ir::Module::new(loc);
        assert!(super::hugr_to_mlir(loc, &h).is_ok());
        assert!(m.as_operation().verify());
    }
    #[test]
    fn test_cfg_hugr() -> super::Result<()> {
        let ctx = get_test_context();
        let h = get_example_hugr_cfg().unwrap();
        let loc = melior::ir::Location::unknown(&ctx);
        let r = super::hugr_to_mlir(loc, &h)?;
        println!("{}", r.as_operation());
        println!("dougrulx");
        assert!(r.as_operation().verify());
        println!("{}", r.as_operation());
        Ok(())
    }
}
