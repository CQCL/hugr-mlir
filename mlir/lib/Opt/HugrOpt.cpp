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

void hugr_mlir::registerHugrOptPipelines() {
  mlir::PassPipelineRegistration<>(
      "lower-hugr", "pipeline to lower hugr", [](mlir::OpPassManager& pm) {
        {
          auto& module_pm = pm.nest<hugr_mlir::ModuleOp>();

          module_pm.addPass(createPreConvertHugrFuncPass());

          module_pm.addPass(mlir::createCanonicalizerPass());
          module_pm.addPass(mlir::createSCCPPass());
          module_pm.addPass(mlir::createCanonicalizerPass());
          module_pm.addPass(mlir::createCSEPass());
          module_pm.addPass(mlir::createCanonicalizerPass());
        }

        pm.addPass(createConvertHugrFuncPass());
        pm.addPass(createConvertHugrPass());
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
