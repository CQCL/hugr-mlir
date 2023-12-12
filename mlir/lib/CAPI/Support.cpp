#include "hugr-mlir-c/Support.h"

#include <assert.h>

#include <functional>

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/Operation.h"

unsigned mlirOperationHash(MlirOperation op) {
  assert(!mlirOperationIsNull(op) && "Op must not be null");
  return mlir::DenseMapInfo<mlir::Operation*>::getHashValue(unwrap(op));
}

unsigned mlirTypeHash(MlirType t) {
  assert(!mlirTypeIsNull(t) && "Type must not be null");
  return mlir::DenseMapInfo<mlir::Type>::getHashValue(unwrap(t));
}

unsigned mlirAttributeHash(MlirAttribute attr) {
  assert(!mlirAttributeIsNull(attr) && "Attribute must not be null");
  return mlir::DenseMapInfo<mlir::Attribute>::getHashValue(unwrap(attr));
}

unsigned mlirBlockHash(MlirBlock b) {
  assert(!mlirBlockIsNull(b) && "Block must not be null");
  return mlir::DenseMapInfo<mlir::Block*>::getHashValue(unwrap(b));
}

unsigned mlirRegionHash(MlirRegion r) {
  assert(!mlirRegionIsNull(r) && "Region must not be null");
  return mlir::DenseMapInfo<mlir::Region*>::getHashValue(unwrap(r));
}

unsigned mlirValueHash(MlirValue v) {
  assert(!mlirValueIsNull(v) && "Value must not be null");
  return mlir::DenseMapInfo<mlir::Value>::getHashValue(unwrap(v));
}
