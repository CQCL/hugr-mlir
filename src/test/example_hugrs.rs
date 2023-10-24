use crate::mlir::test::test_context;
use crate::{Error, Result};
use hugr::builder::{
    BuildError, CFGBuilder, Container, Dataflow, DataflowSubContainer, FunctionBuilder,
    HugrBuilder, ModuleBuilder, SubContainer, TailLoopBuilder,
};
use hugr::extension::prelude::ConstUsize;
use hugr::extension::{
    prelude,
    prelude::{PRELUDE_ID, USIZE_T},
    ExtensionSet,
};
use hugr::hugr::ValidationError;
use hugr::ops::Const;
use hugr::types::{FunctionType, Type};
use hugr::{type_row, Hugr};
use rstest::rstest;

pub(super) const NAT: hugr::types::Type = USIZE_T;
pub(super) const BIT: hugr::types::Type = USIZE_T;

pub fn simple_recursion() -> Result<Hugr, BuildError> {
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

        let sum = entry_b.make_tuple_sum(1, sum2_variants, [inw])?;
        entry_b.finish_with_outputs(sum, [])?
    };
    let mut middle_b =
        cfg_builder.simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?;
    let middle = {
        let c = middle_b.add_load_const(hugr::ops::Const::unary_unit_sum(), ExtensionSet::new())?;
        let [inw] = middle_b.input_wires_arr();
        middle_b.finish_with_outputs(c, [inw])?
    };
    let exit = cfg_builder.exit_block();
    cfg_builder.branch(&entry, 0, &middle)?;
    cfg_builder.branch(&middle, 0, &exit)?;
    cfg_builder.branch(&entry, 1, &exit)?;
    Ok(())
}

pub fn cfg() -> std::result::Result<Hugr, BuildError> {
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

pub fn basic_loop() -> Result<Hugr> {
    let mut loop_b = TailLoopBuilder::new(vec![], vec![BIT], vec![USIZE_T])?;
    let [i1] = loop_b.input_wires_arr();
    let const_wire = loop_b.add_load_const(
        ConstUsize::new(1).into(),
        ExtensionSet::singleton(&PRELUDE_ID),
    )?;

    let break_wire = loop_b.make_break(loop_b.loop_signature()?.clone(), [const_wire])?;
    loop_b.set_outputs(break_wire, [i1])?;
    loop_b.finish_prelude_hugr().map_err(Into::into)
}

pub fn loop_with_conditional() -> Result<Hugr> {
    let mut module_builder = ModuleBuilder::new();
    let mut fbuild = module_builder.define_function(
        "main",
        FunctionType::new(type_row![BIT], type_row![NAT])
            .with_input_extensions(ExtensionSet::singleton(&PRELUDE_ID)),
    )?;
    let _fdef = {
        let [b1] = fbuild.input_wires_arr();
        let loop_id = {
            let mut loop_b = fbuild.tail_loop_builder(vec![(BIT, b1)], vec![], type_row![NAT])?;
            let signature = loop_b.loop_signature()?.clone();
            let const_val = Const::true_val();
            let const_wire = loop_b.add_load_const(Const::true_val(), ExtensionSet::new())?;
            let lift_node = loop_b.add_dataflow_op(
                hugr::ops::LeafOp::Lift {
                    type_row: vec![const_val.const_type().clone()].into(),
                    new_extension: PRELUDE_ID,
                },
                [const_wire],
            )?;
            let [const_wire] = lift_node.outputs_arr();
            let [b1] = loop_b.input_wires_arr();
            let conditional_id = {
                let predicate_inputs = vec![type_row![]; 2];
                let output_row = loop_b.internal_output_row()?;
                let mut conditional_b = loop_b.conditional_builder(
                    (predicate_inputs, const_wire),
                    vec![(BIT, b1)],
                    output_row,
                    ExtensionSet::new(),
                )?;

                let mut branch_0 = conditional_b.case_builder(0)?;
                let [b1] = branch_0.input_wires_arr();

                let continue_wire = branch_0.make_continue(signature.clone(), [b1])?;
                branch_0.finish_with_outputs([continue_wire])?;

                let mut branch_1 = conditional_b.case_builder(1)?;
                let [_b1] = branch_1.input_wires_arr();

                let wire = branch_1.add_load_const(
                    ConstUsize::new(2).into(),
                    ExtensionSet::singleton(&PRELUDE_ID),
                )?;
                let break_wire = branch_1.make_break(signature, [wire])?;
                branch_1.finish_with_outputs([break_wire])?;

                conditional_b.finish_sub_container()?
            };
            loop_b.finish_with_outputs(conditional_id.out_wire(0), [])?
        };
        fbuild.finish_with_outputs(loop_id.outputs())?
    };
    module_builder.finish_prelude_hugr().map_err(Into::into)
}
