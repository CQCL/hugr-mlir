#include "hugr-mlir/IR/HugrAttrs.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "hugr-mlir/IR/HugrAttrs.cpp.inc"

void hugr_mlir::HugrDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hugr-mlir/IR/HugrAttrs.cpp.inc"
      >();
}
