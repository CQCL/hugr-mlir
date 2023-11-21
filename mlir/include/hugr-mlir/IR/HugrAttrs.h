#ifndef HUGR_MLIR_IR_HUGR_ATTRS_H
#define HUGR_MLIR_IR_HUGR_ATTRS_H

#include "hugr-mlir/IR/HugrEnums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class TupleType;
}

#define GET_ATTRDEF_CLASSES
#include "hugr-mlir/IR/HugrAttrs.h.inc"

#endif
