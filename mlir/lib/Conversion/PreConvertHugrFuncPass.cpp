#include "hugr-mlir/Conversion/PreConvertHugrFuncPass.h"

#include "hugr-mlir/Conversion/Utils.h"
#include "hugr-mlir/Transforms/Utils.h"

#include "hugr-mlir/IR/HugrOps.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

namespace hugr_mlir {
#define GEN_PASS_DEF_PRECONVERTHUGRFUNCPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir

namespace {

using namespace mlir;

struct PreConvertHugrFuncPass
    : hugr_mlir::impl::PreConvertHugrFuncPassBase<PreConvertHugrFuncPass> {
  using PreConvertHugrFuncPassBase::PreConvertHugrFuncPassBase;
  // LogicalResult initialize(MLIRContext*) override;
  void runOnOperation() override;
};

struct ClosureiseCallOp : hugr_mlir::FuncClosureMapOpConversionPatternBase<hugr_mlir::CallOp> {
  using FuncClosureMapOpConversionPatternBase::FuncClosureMapOpConversionPatternBase;
  LogicalResult matchAndRewrite(hugr_mlir::CallOp, OpAdaptor, ConversionPatternRewriter&) const override;
};

struct ClosureiseLoadConstantOp : hugr_mlir::FuncClosureMapOpConversionPatternBase<hugr_mlir::LoadConstantOp> {
  using FuncClosureMapOpConversionPatternBase::FuncClosureMapOpConversionPatternBase;
  LogicalResult matchAndRewrite(hugr_mlir::LoadConstantOp, OpAdaptor, ConversionPatternRewriter&) const override;
};

struct ClosureiseConstantOp : hugr_mlir::FuncClosureMapOpConversionPatternBase<hugr_mlir::ConstantOp> {
  using FuncClosureMapOpConversionPatternBase::FuncClosureMapOpConversionPatternBase;
  LogicalResult matchAndRewrite(hugr_mlir::ConstantOp, OpAdaptor, ConversionPatternRewriter&) const override;
};

struct LowerSwitch : OpConversionPattern<hugr_mlir::SwitchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(hugr_mlir::SwitchOp, OpAdaptor, ConversionPatternRewriter&) const override;
};

}

mlir::LogicalResult LowerSwitch::matchAndRewrite(
    hugr_mlir::SwitchOp op, OpAdaptor, ConversionPatternRewriter& rw) const {
  assert(op.getDestinations().size() > 0 && "must");

  auto loc = op.getLoc();
  auto pred = op.getPredicate();
  auto pred_ty = pred.getType();

  SmallVector<int32_t> case_values;
  for (auto i = 0; i < pred_ty.numAlts(); ++i) {
    case_values.push_back(i);
  }

  SmallVector<Block*> case_destinations{op.getDestinations()};
  SmallVector<SmallVector<Value>> case_operands;

  rw.setInsertionPoint(op);
  auto tag = rw.createOrFold<index::CastUOp>(
      loc, rw.getI32Type(), rw.createOrFold<hugr_mlir::ReadTagOp>(loc, pred));

  for (auto [i, t] : llvm::enumerate(
           llvm::cast<hugr_mlir::SumType>(pred.getType()).getTypes())) {
    auto tt = llvm::cast<TupleType>(t);
    auto v = rw.createOrFold<hugr_mlir::ReadVariantOp>(loc, pred, i);
    auto& case_ops = case_operands.emplace_back();
    rw.createOrFold<hugr_mlir::UnpackTupleOp>(case_ops, loc, tt.getTypes(), v);
    llvm::copy(op.getOtherInputs(), std::back_inserter(case_ops));
  }

  assert(
      case_values.size() == case_destinations.size() &&
      case_values.size() == case_operands.size() && case_values.size() > 0 &&
      "must");

  SmallVector<ValueRange> case_operands_vrs;
  llvm::transform(
      case_operands, std::back_inserter(case_operands_vrs),
      [](SmallVectorImpl<Value>& x) { return ValueRange(x); });
  rw.replaceOpWithNewOp<cf::SwitchOp>(
      op, tag, case_destinations[0], case_operands_vrs[0],
      ArrayRef(case_values).drop_front(),
      ArrayRef(case_destinations).drop_front(),
      ArrayRef(case_operands_vrs).drop_front());
  return success();
}


LogicalResult ClosureiseConstantOp::matchAndRewrite(
    hugr_mlir::ConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rw) const {
  auto a = llvm::dyn_cast<hugr_mlir::StaticEdgeAttr>(op.getValue());
  if (!a || !llvm::isa<hugr_mlir::FunctionType>(a.getType())) {
    return failure();
  }
  auto alloc_op = lookupAllocFunctionOp(a.getRef());
  if (!alloc_op) {
    return failure();
  }
  rw.setInsertionPoint(op);
  rw.replaceOp(op, rw.getRemappedValue(alloc_op.getOutput()));
  return success();
}

LogicalResult ClosureiseLoadConstantOp::matchAndRewrite(
    hugr_mlir::LoadConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rw) const {
  auto a = op.getConstRef();
  if (!a || !llvm::isa<hugr_mlir::FunctionType>(a.getType())) {
    return failure();
  }

  rw.setInsertionPoint(op);
  if (auto alloc_op = lookupAllocFunctionOp(a.getRef())) {
    rw.replaceOp(op, rw.getRemappedValue(alloc_op.getOutput()));
  } else if (auto func_op = lookupTopLevelFunc(a.getRef())){
    auto func = rw.createOrFold<func::ConstantOp>(op.getLoc(), getTypeConverter()->convertType(a.getType()), func_op.getSymName());
    auto closure = rw.createOrFold<ub::PoisonOp>(op.getLoc(), rw.getType<hugr_mlir::ClosureType>().getMemRefType(), rw.getAttr<ub::PoisonAttr>());
    rw.replaceOp(op, SmallVector{func,closure});
  } else {
    return failure();
  }
  return success();
}

LogicalResult ClosureiseCallOp::matchAndRewrite(
    hugr_mlir::CallOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rw) const {
  auto attr = op.getCalleeAttrAttr();
  if (!attr) {
    return failure();
  }

  auto alloc_op = lookupAllocFunctionOp(attr.getRef());
  if (!alloc_op) {
    return failure();
  }

  rw.replaceOpWithNewOp<hugr_mlir::CallOp>(op, alloc_op.getOutput(), op.getInputs());
  return mlir::success();
}

void PreConvertHugrFuncPass::runOnOperation() {
  auto context = &getContext();
  // collect funcs
  llvm::SmallVector<hugr_mlir::FuncOp> hugr_funcs;
  getOperation()->walk<WalkOrder::PostOrder>(
      [&hugr_funcs](hugr_mlir::FuncOp op) { hugr_funcs.push_back(op); });

  // alloc_closure at the region entry block for each func, map sym -> value
  hugr_mlir::FuncClosureMap func_closures;
  IRRewriter rw(context);
  for (auto f : hugr_funcs) {
    if (isTopLevelFunc(f)) {
      if(failed(func_closures.insert(f))) {
        emitError(f.getLoc()) << "PreConvertHugrPass:Failed to collect top level func";
        return signalPassFailure();
      }
    } else {
      rw.setInsertionPointToStart(&f->getParentRegion()->front());
      auto o = rw.create<hugr_mlir::AllocFunctionOp>(
          f.getLoc(), f.getStaticEdgeAttr());
      if(failed(func_closures.insert(o))) {
        emitError(f.getLoc()) << "PreConvertHugrPass:Failed to collect func";
        return signalPassFailure();
      }
    }
  }

  // legalize calls to use a closure
  ConversionTarget target(*context);
  target.addLegalDialect<
      cf::ControlFlowDialect, hugr_mlir::HugrDialect, index::IndexDialect,
      func::FuncDialect>();
  target.addIllegalOp<hugr_mlir::SwitchOp>();
  target.addDynamicallyLegalOp<hugr_mlir::CallOp>([&](hugr_mlir::CallOp op) {

    return op.getCalleeValue() ||
           func_closures.lookupTopLevelFunc(op.getCalleeAttrAttr().getRef());

  });
  target.addDynamicallyLegalOp<hugr_mlir::LoadConstantOp>(
      [&](hugr_mlir::LoadConstantOp op) {
        return !llvm::isa<hugr_mlir::FunctionType>(op.getConstRef().getType());
      });
  target.addDynamicallyLegalOp<hugr_mlir::ConstantOp>(
      [&](hugr_mlir::ConstantOp op) {
        return !llvm::isa<hugr_mlir::FunctionType>(op.getValue().getType()) ||
               !!func_closures.lookupTopLevelFunc(
                   llvm::cast<hugr_mlir::StaticEdgeAttr>(op.getValue())
                       .getRef()
                       .getLeafReference());
      });

  {
    RewritePatternSet ps(context);
    ps.add<ClosureiseCallOp, ClosureiseConstantOp, ClosureiseLoadConstantOp>(
        func_closures, context);
    ps.add<LowerSwitch>(context);
    FrozenRewritePatternSet patterns(
        std::move(ps), disabledPatterns, enabledPatterns);
    if (failed(applyPartialConversion(getOperation(), target, patterns))) {
      emitError(getOperation()->getLoc()) << "PreConvertHugrPass: Failed to closureize call ops";
      return signalPassFailure();
    }
  }

  // legalize funcs to have no non-local edges
  for (auto f : hugr_funcs) {
    if (f.isDeclaration() || isTopLevelFunc(f)) {
      continue;
    }

    auto orig_num_caps = f.getCaptures().size();
    llvm::SetVector<Value> captures;
    for (auto v : f.getCaptures()) {
      captures.insert(v);
    }
    getUsedValuesDefinedAbove(f.getBody(), captures);

    rw.updateRootInPlace(f, [&] {
      auto old_launch_block = &f.getBody().front();
      SmallVector<Type> new_launch_tys{TypeRange{captures.getArrayRef()}};
      SmallVector<Location> new_launch_locs;
      llvm::transform(
          captures, std::back_inserter(new_launch_locs),
          [](auto x) { return x.getLoc(); });
      for (auto ba :
           old_launch_block->getArguments().drop_front(orig_num_caps)) {
        new_launch_tys.push_back(ba.getType());
        new_launch_locs.push_back(ba.getLoc());
      }

      auto new_launch_block =
          rw.createBlock(old_launch_block, new_launch_tys, new_launch_locs);
      SmallVector<Value> replacements(orig_num_caps);
      for (auto i = 0u; i < orig_num_caps; ++i) {
        auto it = llvm::find(captures, f.getCaptures()[i]);
        assert(it != captures.end() && "we must have all of f's old captures");
        replacements[i] = new_launch_block->getArgument(it - captures.begin());
      }
      llvm::copy(
          new_launch_block->getArguments().drop_front(captures.size()),
          std::back_inserter(replacements));
      rw.mergeBlocks(old_launch_block, new_launch_block, replacements);
      f.getCapturesMutable().assign(captures.getArrayRef());
      rw.replaceUsesWithIf(
          captures.getArrayRef(),
          new_launch_block->getArguments().take_front(captures.size()),
          [&](OpOperand& oo) { return f->isProperAncestor(oo.getOwner()); });
      assert(
          !hugr_mlir::opHasNonLocalEdges(f) &&
          "we just removed nonlocal edges");
    });
  }
  target.addDynamicallyLegalOp<hugr_mlir::FuncOp>(
      [](hugr_mlir::FuncOp op) { return !hugr_mlir::opHasNonLocalEdges(op); });

  // TODO this can be disabled in production
  if (failed(applyPartialConversion(getOperation(), target, {}))) {
    emitError(getOperation()->getLoc())
        << "Failed to legalize funcs to have no non-local edges";
    return signalPassFailure();
  }

  //
  // canoncialize + sccp
  //
  // hugr.func -> func.func(out of line, taking closure as arg) + write_closure
}
