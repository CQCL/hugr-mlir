// clang-format off
#include "hugr-mlir/IR/HugrDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "hugr-mlir/IR/HugrOps.h"

#include "hugr-mlir/IR/HugrDialect.cpp.inc"
// clang-format on


void hugr_mlir::HugrDialect::initialize() {
  registerTypes();
  registerAttrs();
  registerOps();
  registerTypeInterfaces();
}
