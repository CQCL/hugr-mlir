#include "hugr-mlir/IR/HugrTypes.h"

#include "hugr-mlir/IR/HugrAttrs.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrTypeInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "hugr-mlir/IR/HugrTypes.cpp.inc"

void hugr_mlir::HugrDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hugr-mlir/IR/HugrTypes.cpp.inc"
      >();
}
