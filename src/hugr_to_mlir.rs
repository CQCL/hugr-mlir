use std::{borrow::Borrow, iter::zip};
use std::ops::Deref;

use hugr::HugrView;
use hugr::types::EdgeKind;
use melior::{ir::BlockRef, Context};
use std::collections::HashMap;
use std::vec::Vec;

use crate::{mlir, Error};

pub mod types;
use types::*;

use melior::ir::Value;

type Scope<'a, 'b> = HashMap<(hugr::Node, hugr::Port), Value<'a, 'b>>;

struct TranslationState<'a, 'b, V> {
    context: &'a melior::Context,
    hugr: &'a V,
    scope:Scope<'a, 'b>,
    block: &'b melior::ir::Block<'a>,
}

// fn last_in_block<'c,'b>(block: &'b melior::ir::Block<'c>) -> Option<&'b melior::ir::Operation<'c>>
//     where 'c: 'b
// {
//     let get_next = |x: &'b melior::ir::Operation<'c>| x.next_in_block().map(|y| unsafe {y.to_ref()});
//     let mut best : Option<&'b melior::ir::Operation<'c>> = block.first_operation().map(|y: melior::ir::OperationRef<'c, 'b>| unsafe { y.to_ref() });
//     while best.is_some() {
//         best = best.and_then(get_next);
//     }
//     best
// }

pub fn push_scope<'a,'b>(mut scope: Scope<'a,'b>, binders: impl IntoIterator<Item= <Scope<'a,'b> as IntoIterator>::Item>) -> Scope<'a,'b> {
    for (k,v) in binders {
        scope.insert(k,v);
    }
    scope
}


impl<'a, 'b, V: HugrView> TranslationState<'a, 'b, V>
where
    'a: 'b,
{
    fn new(context: &'a melior::Context, hugr: &'a V, block: &'b melior::ir::Block<'a>) -> Self {
        TranslationState {
            context,
            hugr,
            block,
            scope: HashMap::new(),
        }
    }

    fn push_block<'c>(
        &'c self,
        block: &'c  melior::ir::Block<'a>,
        block_args: impl
            IntoIterator<Item =(hugr::Node,hugr::Port)>
    ) -> Result<TranslationState<'a, 'c, V>, Error>
    {
        let scope = push_scope(self.scope.clone(), block_args.into_iter().enumerate().map(|(i, k)| (k, block.argument(i).unwrap().into())));
        Ok(TranslationState { context: self.context, hugr: self.hugr, block, scope})
    }

    fn lookup_nodeport(
        &'_ self,
        n: hugr::Node,
        p: hugr::Port,
    ) -> Value<'a, 'b> {
        *self.scope.get(&(n, p)).unwrap_or_else(|| panic!("lookup_nodeport: {:?}\n{:?}", (n,p), &self.scope))
    }

    fn node_to_op(mut self, n: hugr::Node) -> Result<Self, Error>
    {
        use hugr::ops::OpType;
        let ul = melior::ir::Location::unknown(self.context);
        let optype = self.hugr.get_optype(n);
        let mut inputs_port_val: Vec<(hugr::Port, Value<'a, 'b>)> = Vec::new();
        for p in self.hugr.node_inputs(n) {
            match optype.port_kind(p) {
                Some(EdgeKind::Value { .. }) => match self.hugr.linked_ports(n, p).next() {
                    Some((m,q)) => Ok(Some(inputs_port_val.push((p, self.lookup_nodeport(m,q))))),
                    None => Err(Error::StringError("Failed to find linked port".into()))
                }?, _ => None
            };
        }
        let (op, nodeports): (melior::ir::operation::Operation<'a>,_) = match optype {
            &OpType::Module(_) => {
                let body = melior::ir::Region::new();
                let block = body.append_block(melior::ir::Block::new(&[]));
                let mut inner_state = self.push_block(block.deref(), [])?;
                for c in self.hugr.children(n) {
                    inner_state = inner_state.node_to_op(c)?;
                }
                Ok((mlir::hugr::ModuleOp::new(body, ul).into(), Vec::new()))
            }
            &OpType::FuncDefn(hugr::ops::FuncDefn {
                ref name,
                ref signature,
            }) => {
                let name_attr = melior::ir::attribute::StringAttribute::new(self.context, &name);
                let type_ = hugr_to_mlir_function_type(self.context, signature)?;
                let body = self.build_dataflow_region(n)?;
                Ok((
                    mlir::hugr::FuncOp::new(body, name_attr, type_, ul).into(),
                    Vec::new(),
                ))
            }
            &OpType::Call(hugr::ops::Call { ref signature }) => {
                let fun_name = self
                    .hugr
                    .all_node_ports(n)
                    .find_map(|p| match optype.port_kind(p) {
                        Some(EdgeKind::Static { .. }) => {
                            self.hugr.linked_ports(n, p).find_map(|(fun, _)| {
                                match self.hugr.get_optype(fun) {
                                    &OpType::FuncDecl(hugr::ops::FuncDecl { ref name, .. })
                                    | &OpType::FuncDefn(hugr::ops::FuncDefn { ref name, .. }) => {
                                        Some(name)
                                    }, &_ => None
                                }
                            })
                        }
                        _ => None,
                    })
                    .ok_or(Error::StringError(
                        "Failed to find static edge to function".into(),
                    ))?;
                let static_edge = mlir::hugr::StaticEdgeAttr::new(
                    hugr_to_mlir_function_type(self.context, signature)?.into(),
                    melior::ir::attribute::FlatSymbolRefAttribute::new(self.context, fun_name)
                        .into(),
                );

                let results_port_type = self
                    .hugr
                    .node_outputs(n)
                    .filter_map(|p| match optype.port_kind(p) {
                        Some(EdgeKind::Value(ref t)) => {
                            Some(hugr_to_mlir_type(self.context, t).map(|x| (p, x)))
                        }
                        _ => None,
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let inputs: Vec<_> = inputs_port_val.into_iter().map(|(_, v)| v.into()).collect();
                let result_types: Vec<_> =
                    results_port_type.iter().map(|(_, t)| t.clone()).collect();
                let call = mlir::hugr::CallOp::new(
                    static_edge,
                    inputs.as_slice(),
                    result_types.as_slice(),
                    ul,
                );
                Ok::<_,Error>((
                    call.into(),
                    results_port_type.into_iter().map(|(p, _)| (n, p)).collect(),
                ))
            }
            t => panic!("unimplemented: {:?}", t),
        }?;
        let op_ref: melior::ir::OperationRef<'a,'b> = self.block.append_operation(op);
        self.scope = push_scope(self.scope, zip(nodeports, unsafe {op_ref.to_ref()}.results().map(Into::into)));
        Ok(self)
    }

    fn build_dataflow_region(&self, parent: hugr::Node) -> Result<melior::ir::Region<'a>, Error> {
        let ul = melior::ir::Location::unknown(self.context);
        let [i, o] = self
            .hugr
            .get_io(parent)
            .ok_or("FuncDefn has no io nodes".to_string())?;

        let input_type = self.hugr.get_optype(i);
        let output_type = self.hugr.get_optype(o);
        let body: melior::ir::Region<'a> = melior::ir::Region::new();
        let block_arg_port_type_loc = self
            .hugr
            .node_outputs(i)
            .filter_map(move |p| match input_type.port_kind(p) {
                Some(hugr::types::EdgeKind::Value(t)) => Some((p, t)),
                _ => None,
            })
            .map(|(p, ref t)| Ok((p, hugr_to_mlir_type(self.context, t)?, ul)))
            .collect::<Result<Vec<(hugr::Port,melior::ir::Type<'a>, melior::ir::Location<'a>)>,Error>>()?;
        let block = body.append_block(melior::ir::Block::new(
            block_arg_port_type_loc
                .iter()
                .map(|(_, t, l)| (*t, *l))
                .collect::<Vec<_>>()
                .as_slice(),
        ));
        let mut inner_state = self.push_block(block.deref(), block_arg_port_type_loc.iter().map(|(p,_,_)| (i,*p)))?;
        for c in self.hugr.children(parent).filter(|x| *x != i && *x != o) {
            inner_state = inner_state.node_to_op(c)?;
        }
        let output_values = self
            .hugr
            .node_inputs(o)
            .filter_map(|p| match output_type.port_kind(p) {
                Some(EdgeKind::Value{..}) => Some(match self
                    .hugr
                    .linked_ports(o, p)
                    .collect::<std::vec::Vec<_>>()
                    .as_slice()
                {
                    [(m, x)] => Ok(inner_state.lookup_nodeport(*m, *x)),
                    _ => Err(Error::StringError(
                        "Not a unique link to output port".to_string(),
                    )),
                }).map(Into::into),
                _ => None
            })
            .collect::<Result<Vec<_>, Error>>()?;
        block.append_operation(mlir::hugr::OutputOp::new(&output_values, ul).into());
        Ok(body)
    }
}

pub fn hugr_to_mlir(
    ctx: &Context,
    hugr: &impl HugrView,
    block: melior::ir::BlockRef,
) -> Result<(), Error> {
    let mut state = TranslationState::new(ctx, hugr, block.deref());
    for n in hugr.children(hugr.root()) {
        state = state.node_to_op(n)?;
    }
    Ok(())
}

#[cfg(test)]
mod test {

    use crate::mlir::test::get_test_context;

    fn get_example_hugr() -> Result<hugr::Hugr, hugr::builder::BuildError> {
        use hugr::builder::{Dataflow, DataflowSubContainer, HugrBuilder};
        const NAT: hugr::types::Type = hugr::extension::prelude::USIZE_T;
        let mut module_builder = hugr::builder::ModuleBuilder::new();

        let f_id = module_builder.declare(
            "main",
            hugr::types::FunctionType::new(hugr::type_row![NAT], hugr::type_row![NAT]).pure(),
        )?;

        let mut f_build = module_builder.define_declaration(&f_id)?;
        let call = f_build.call(&f_id, f_build.input_wires())?;

        f_build.finish_with_outputs(call.outputs())?;
        module_builder.finish_prelude_hugr().map_err(|x| x.into())
    }

    #[test]
    fn test_example_hugr() {
        let ctx = get_test_context();
        let h = get_example_hugr().unwrap();
        let m = melior::ir::Module::new(melior::ir::Location::unknown(&ctx));
        assert_eq!(super::hugr_to_mlir(&ctx, &h, m.body()), Ok(()));
        assert!(m.as_operation().verify());
    }
}
