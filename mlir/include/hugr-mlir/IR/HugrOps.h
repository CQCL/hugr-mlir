#ifndef HUGR_MLIR_IR_HUGR_OPS_H
#define HUGR_MLIR_IR_HUGR_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#define GET_OP_CLASSES
#include "hugr-mlir/IR/HugrOps.h.inc"

#endif  // HUGR_MLIR_IR_HUGR_OPS_H
