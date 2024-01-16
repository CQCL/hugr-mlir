#include "hugr-mlir/Opt/HugrOpt.h"

#include "hugr-mlir/Analysis/Passes.h"
#include "hugr-mlir/Conversion/Passes.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"
#include "hugr-mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

void hugr_mlir::registerHugrOptPipelines() {
  mlir::PassPipelineRegistration<>(
      "lower-hugr", "pipeline to lower hugr", [](mlir::OpPassManager& module_pm) {
        {
          auto& pm = module_pm.nest<hugr_mlir::ModuleOp>();
          pm.addPass(createPreConvertHugrFuncPass());
          pm.addPass(createConvertHugrExtArithPass());
        }
        {
          auto& pm = module_pm;
          pm.addPass(createConvertHugrFuncPass());
          pm.addPass(createConvertHugrPass());
        }
        {
          auto& pm = module_pm.nest<mlir::func::FuncOp>();
          pm.addPass(mlir::createCanonicalizerPass());
        }
        {
          auto& pm = module_pm;
          pm.addPass(mlir::createSCCPPass());
        }
        {
          auto& pm = module_pm.nest<mlir::func::FuncOp>();
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
          pm.addPass(mlir::createCanonicalizerPass());
        }
        {
          auto& pm = module_pm;
          pm.addPass(mlir::createSymbolDCEPass());
          pm.addPass(hugr_mlir::createFinalizeHugrToLLVMPass());
        }
        {
          auto& pm = module_pm.nest<mlir::LLVM::LLVMFuncOp>();
          pm.addPass(mlir::createCanonicalizerPass());
          // TODO infer data layout
        }
      });

}

int hugr_mlir::HugrMlirOptMain(
    int argc, char** argv, mlir::DialectRegistry* registry) {
  mlir::DialectRegistry local_registry;
  if (!registry) {
    registry = &local_registry;
  }

  registerHugrAnalysisPasses();
  registerHugrTransformsPasses();
  registerHugrConversionPasses();
  mlir::registerTransformsPasses();
  registerHugrOptPipelines();

  registry->insert<
      hugr_mlir::HugrDialect, mlir::arith::ArithDialect,
      mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
      mlir::LLVM::LLVMDialect, mlir::index::IndexDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "hugr mlir optimizer driver\n", *registry));
}
