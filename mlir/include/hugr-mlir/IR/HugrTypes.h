#ifndef HUGR_MLIR_IR_HUGR_TYPES_H
#define HUGR_MLIR_IR_HUGR_TYPES_H

#include "hugr-mlir/IR/HugrAttrs.h"
#include "mlir/IR/TypeRange.h"

namespace hugr_mlir {
class ExtensionSetAttr;
class HugrTypeInterface;
}  // namespace hugr_mlir

#define GET_TYPEDEF_CLASSES
#include "hugr-mlir/IR/HugrTypes.h.inc"

namespace hugr_mlir {

bool areTypesCompatible(mlir::Type, mlir::Type);
bool areTypesCompatible(mlir::TypeRange, mlir::TypeRange);

}  // namespace hugr_mlir

#endif
