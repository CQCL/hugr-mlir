#include "hugr-mlir/IR/HugrDialect.h"

#include "hugr-mlir/IR/HugrDialect.cpp.inc"
#include "hugr-mlir/IR/HugrOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

void hugr_mlir::HugrDialect::initialize() {
  registerTypes();
  registerAttrs();
  registerOps();
  registerTypeInterfaces();
}
