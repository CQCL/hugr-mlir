//===- HugrAttrs.td - Hugr dialect attributes -------*- tablegen -*-===//

#ifndef HUGR_MLIR_IR_HUGR_ATTRS_TD
#define HUGR_MLIR_IR_HUGR_ATTRS_TD

include "hugr-mlir/IR/HugrDialect.td"
include "hugr-mlir/IR/HugrTypes.td"
include "hugr-mlir/IR/HugrEnums.td"

def Hugr_TypeConstraintAttr : Hugr_EnumAttr<Hugr_TypeConstraintEnum, "constraint", []> {
    let summary = "The set of constraints on a type";
    let assemblyFormat = "`<`$value`>`";
    let extraClassDeclaration = [{
        TypeConstraintAttr intersection(TypeConstraintAttr);
    }];
}

#endif
