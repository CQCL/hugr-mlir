#include "hugr-mlir/IR/HugrOps.h"

#include "hugr-mlir/IR/HugrAttrs.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrTypeInterfaces.h"
#include "hugr-mlir/IR/HugrTypes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "hugr-mlir/IR/HugrOps.cpp.inc"

void hugr_mlir::HugrDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "hugr-mlir/IR/HugrOps.cpp.inc"
      >();
}
