#include "hugr-mlir/Conversion/FinalizeHugrToLLVMPass.h"

#include "hugr-mlir/IR/HugrTypes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace hugr_mlir {
#define GEN_PASS_DEF_FINALIZEHUGRTOLLVMPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir

static mlir::LowerToLLVMOptions getLowerToLLVMOptions(mlir::MLIRContext* context) {
  auto r = mlir::LowerToLLVMOptions(context);
  r.useBarePtrCallConv = true;
  return r;
}

namespace {
using namespace mlir;



struct HugrLLVMTypeConverter : LLVMTypeConverter {
  HugrLLVMTypeConverter(MLIRContext* context) : LLVMTypeConverter(context, getLowerToLLVMOptions(context)) {
    addConversion([this](hugr_mlir::ClosureType t) {
      return LLVM::LLVMPointerType::get(t.getContext());
    });
  }
};

struct FinalizeHugrToLLVMPass
    : hugr_mlir::impl::FinalizeHugrToLLVMPassBase<FinalizeHugrToLLVMPass> {
  using FinalizeHugrToLLVMPassBase::FinalizeHugrToLLVMPassBase;
  LogicalResult initialize(MLIRContext*) override;
  void runOnOperation() override;
private:
  std::shared_ptr<HugrLLVMTypeConverter> type_converter;
  FrozenRewritePatternSet patterns;

};

}

LogicalResult FinalizeHugrToLLVMPass::initialize(MLIRContext * context) {
  type_converter = std::make_shared<HugrLLVMTypeConverter>(context);
  RewritePatternSet ps(context);
  populateFinalizeMemRefToLLVMConversionPatterns(*type_converter, ps);
  arith::populateArithToLLVMConversionPatterns(*type_converter, ps);
  populateMathToLLVMConversionPatterns(*type_converter, ps);
  index::populateIndexToLLVMConversionPatterns(*type_converter, ps);
  ub::populateUBToLLVMConversionPatterns(*type_converter, ps);
  cf::populateControlFlowToLLVMConversionPatterns(*type_converter, ps);
  populateFuncToLLVMConversionPatterns(*type_converter, ps);

  patterns = FrozenRewritePatternSet(std::move(ps), disabledPatterns, enabledPatterns);

  return success();
}

void FinalizeHugrToLLVMPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  if(failed(applyFullConversion(getOperation(), target, patterns))) {
    emitError(getOperation().getLoc()) << "FinalizeHugrToLLVMPass: Failed to applyFullConversion";
    return signalPassFailure();
  }
}
