#ifndef HUGR_MLIR_IR_HUGR_ENUMS_TD
#define HUGR_MLIR_IR_HUGR_ENUMS_TD

include "hugr-mlir/IR/HugrDialect.td"

def Hugr_TypeConstraintEnum : I32EnumAttr<
    "TypeConstraint",
    "The type constraint of a Type", [
         I32EnumAttrCase<"Linear", 0>,
         I32EnumAttrCase<"Equatable", 1>,
         I32EnumAttrCase<"Copyable", 2>
    ]> {
  let cppNamespace = Hugr_Dialect.cppNamespace;
  let genSpecializedAttr = 0;
}

#endif
