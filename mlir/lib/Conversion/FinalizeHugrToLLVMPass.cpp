#include "hugr-mlir/Conversion/FinalizeHugrToLLVMPass.h"

#include "hugr-mlir/IR/HugrTypes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "llvm/ADT/TypeSwitch.h"

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
}

static LogicalResult wrangle_main(RewriterBase& rw, func::FuncOp main) {
  if(!llvm::isa_and_present<ModuleOp>(main->getParentOp()) || main.getBody().empty()) { return success(); }

  auto loc = main.getLoc();
  OpBuilder::InsertionGuard _g(rw);

  auto parent = llvm::cast<ModuleOp>(main->getParentOp());
  auto ft = main.getFunctionType();
  if(ft.getInputs() != TypeRange{rw.getType<hugr_mlir::ClosureType>().getMemRefType()} ||
     llvm::any_of(ft.getResults(), [](auto t) {
       return llvm::isa<FloatType,IntegerType>(t);
     })) {
    return emitError(main.getLoc()) << "Invalid function type for main: Expected a single memref<!hugr_mlir.closure> argument and only int or float results";
  }

  auto shim_sym = "_hugr.main";
  rw.updateRootInPlace(main, [&] {
    main.setSymName(shim_sym);
  });
  rw.setInsertionPoint(main);

  auto func = rw.create<func::FuncOp>(main.getLoc(), "main", rw.getType<FunctionType>(TypeRange{},TypeRange{}));
  rw.updateRootInPlace(func, [&] { func.setSymVisibility("public"); });
  auto block = rw.createBlock(&func.getBody(), func.getBody().end());
  rw.setInsertionPointToStart(block);
  auto null = rw.createOrFold<LLVM::NullOp>(loc, rw.getType<LLVM::LLVMPointerType>());
  rw.create<func::CallOp>(loc, "__quantum__rt__initialize", TypeRange{}, null);
  auto closure = rw.createOrFold<ub::PoisonOp>(loc, ft.getInput(0), rw.getAttr<ub::PoisonAttr>());
  auto call = rw.create<func::CallOp>(loc, main, closure);


  for(auto r: call.getResults()) {
    Value v = r;
    if(failed(llvm::TypeSwitch<Type, LogicalResult>(r.getType())
      .Case([&](FloatType t) {
        if(t.getWidth() < 64) {
          v = rw.createOrFold<arith::ExtFOp>(loc, rw.getF64Type(), r);
        } else if (t.getWidth() > 64) {
          v = rw.createOrFold<arith::TruncFOp>(loc, rw.getF64Type(), r);
        }
        return success();
      }).Case([&](IntegerType t) {
        v = rw.createOrFold<arith::UIToFPOp>(loc, rw.getF64Type(), r);
        return success();
      }).Case([&](IndexType t) {
        v = rw.createOrFold<arith::UIToFPOp>(loc, rw.getF64Type(), rw.createOrFold<index::CastUOp>(loc, rw.getI64Type(), r));
        return success();
      }).Default([&](auto t) { return emitError(r.getLoc()) << "Failed to convert type to double: " << t;
      }))) {
      return failure();
    }
    SmallVector<Value> args{v, null};
    rw.create<func::CallOp>(loc, "__quantum__rt__double_record_output", TypeRange{}, args);
  }
  rw.create<func::ReturnOp>(loc);
  if(!parent.lookupSymbol("__quantum__rt__double_record_output")) {
    rw.setInsertionPointToEnd(parent.getBody());
    rw.create<func::FuncOp>(loc, "__quantum__rt__double_record_output", rw.getType<FunctionType>(SmallVector<Type>{rw.getF64Type(), null.getType()}, TypeRange{}));
  }

  if(!parent.lookupSymbol("__quantum__rt__initialize")) {
    rw.setInsertionPointToEnd(parent.getBody());
    rw.create<func::FuncOp>(loc, "__quantum__rt__initialize", rw.getType<FunctionType>(null.getType(), TypeRange{}));
  }
  return success();
}

void FinalizeHugrToLLVMPass::runOnOperation() {
  auto context = &getContext();
  IRRewriter rw(context);
  if(auto main = getOperation().lookupSymbol<func::FuncOp>("main")) {
    if(failed(wrangle_main(rw, main))) {
      return signalPassFailure();
    }
  }


  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  if(failed(applyFullConversion(getOperation(), target, patterns))) {
    emitError(getOperation().getLoc()) << "FinalizeHugrToLLVMPass: Failed to applyFullConversion";
    return signalPassFailure();
  }
}
