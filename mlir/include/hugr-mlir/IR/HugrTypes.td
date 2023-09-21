//===- HugrTypes.td - Hugr dialect types -------*- tablegen -*-===//

#ifndef HUGR_MLIR_IR_HUGR_TYPES_TD
#define HUGR_MLIR_IR_HUGR_TYPES_TD

include "hugr-mlir/IR/HugrDialect.td"

def Hugr_TestType : Hugr_Type<"Test", "test"> {
    let assemblyFormat = "";
}


#endif
