#include "hugr-mlir/IR/HugrDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<
      hugr_mlir::HugrDialect, mlir::arith::ArithDialect,
      mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
      mlir::LLVM::LLVMDialect>();
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
