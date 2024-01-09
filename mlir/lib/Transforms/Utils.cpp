#include "hugr-mlir/Transforms/Utils.h"

#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/RegionUtils.h"

bool hugr_mlir::opHasNonLocalEdges(mlir::Operation* op) {
  llvm::SetVector<mlir::Value> nonlocals;
  mlir::getUsedValuesDefinedAbove(op->getRegions(), nonlocals);
  return !nonlocals.empty();
}
