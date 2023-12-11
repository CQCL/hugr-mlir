#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/Analysis/Passes.h"
#include "hugr-mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  hugr_mlir::registerHugrAnalysisPasses();
  hugr_mlir::registerHugrTransformsPasses();


  mlir::DialectRegistry registry;
  registry.insert<
      hugr_mlir::HugrDialect, mlir::arith::ArithDialect,
      mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
      mlir::LLVM::LLVMDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "hugr mlir optimizer driver\n", registry));
}
