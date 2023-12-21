#include "hugr-mlir/Transforms/LowerHugrPass.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"

// #include "hugr-mlir/Conversion/ConvertHugrToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

namespace hugr_mlir {
#define GEN_PASS_DEF_LOWERHUGRPASS
#include "hugr-mlir/Transforms/Passes.h.inc"
}  // namespace hugr_mlir

namespace {

using namespace mlir;

struct LowerHugrPass : hugr_mlir::impl::LowerHugrPassBase<LowerHugrPass> {
  using LowerHugrPassBase::LowerHugrPassBase;
  LogicalResult initialize(MLIRContext*) override;
  void runOnOperation() override;

 private:
  FrozenRewritePatternSet patterns;
};

struct LowerHugrFuncToFunc : OpRewritePattern<hugr_mlir::FuncOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::FuncOp, PatternRewriter&) const override;
};

struct LowerCfg : OpRewritePattern<hugr_mlir::CfgOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::CfgOp, PatternRewriter&) const override;
};

struct LowerOutput : OpRewritePattern<hugr_mlir::OutputOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::OutputOp, PatternRewriter&) const override;
};

struct LowerSwitch : OpRewritePattern<hugr_mlir::SwitchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::SwitchOp, PatternRewriter&) const override;
};

}  // namespace

mlir::LogicalResult LowerHugrFuncToFunc::matchAndRewrite(
    hugr_mlir::FuncOp op, PatternRewriter& rw) const {
  auto& body = op.getBody();
  if (!body.empty()) {
    SetVector<Value> used_above;
    getUsedValuesDefinedAbove(body, used_above);
    if (!used_above.empty()) {
      return failure();
    }
  }

  rw.setInsertionPoint(op);
  NamedAttrList list(op->getDiscardableAttrs());
  if (auto attr = op.getSymVisibilityAttr()) {
    list.append(
        func::FuncOp::getSymVisibilityAttrName(
            OperationName("func.func", getContext())),
        attr);
  }
  if (auto res_attrs = op.getResAttrsAttr()) {
    list.append(
        func::FuncOp::getResAttrsAttrName(
            OperationName("func.func", getContext())),
        res_attrs);
  }
  if (auto arg_attrs = op.getArgAttrsAttr()) {
    list.append(
        func::FuncOp::getArgAttrsAttrName(
            OperationName("func.func", getContext())),
        arg_attrs);
  }
  auto func = rw.create<func::FuncOp>(
      op.getLoc(), op.getSymName(), op.getFunctionType().getFunctionType(),
      list.getAttrs());
  if (!body.empty()) {
    rw.inlineRegionBefore(body, func.getBody(), func.getBody().end());
  }
  rw.eraseOp(op);
  return success();
}

mlir::LogicalResult LowerCfg::matchAndRewrite(
    hugr_mlir::CfgOp op, PatternRewriter& rw) const {
  Block* parent_block = op->getBlock();
  Region& body = op.getBody();

  if (body.empty() || !parent_block) {
    return failure();
  }

  auto loc = op.getLoc();
  Block* exit_block;
  {
    auto tail_block = rw.splitBlock(parent_block, Block::iterator(op));
    auto output_tys = op.getOutputs().getTypes();
    exit_block = rw.createBlock(
        tail_block, output_tys, SmallVector<Location>(output_tys.size(), loc));
    rw.mergeBlocks(tail_block, exit_block);
  }

  SmallVector<hugr_mlir::OutputOp> outputs{
      op.getBody().getOps<hugr_mlir::OutputOp>()};
  for (auto output : outputs) {
    rw.setInsertionPoint(output);
    rw.replaceOpWithNewOp<cf::BranchOp>(
        output, exit_block, output.getOutputs());
  }

  auto body_entry = &body.front();
  rw.inlineRegionBefore(body, exit_block);
  rw.setInsertionPointToEnd(parent_block);
  rw.create<cf::BranchOp>(loc, body_entry, op.getInputs());
  rw.replaceOp(op, exit_block->getArguments());
  return success();
}

mlir::LogicalResult LowerOutput::matchAndRewrite(
    hugr_mlir::OutputOp op, PatternRewriter& rw) const {
  auto parent = op->getParentOp();
  if (!llvm::isa_and_nonnull<func::FuncOp>(parent)) {
    return failure();
  }
  rw.replaceOpWithNewOp<func::ReturnOp>(op, op.getOutputs());

  return success();
}

mlir::LogicalResult LowerSwitch::matchAndRewrite(
    hugr_mlir::SwitchOp op, PatternRewriter& rw) const {
  assert(op.getDestinations().size() > 0 && "must");

  auto loc = op.getLoc();
  auto pred = op.getPredicate();
  auto pred_ty = llvm::cast<hugr_mlir::SumType>(pred.getType());

  SmallVector<int32_t> case_values;
  for (auto i = 0; i < pred_ty.numAlts(); ++i) {
    case_values.push_back(i);
  }

  SmallVector<Block*> case_destinations{op.getDestinations()};
  SmallVector<SmallVector<Value>> case_operands;

  auto tag = rw.createOrFold<index::CastSOp>(
      loc, rw.getI32Type(), rw.createOrFold<hugr_mlir::ReadTagOp>(loc, pred));

  for (auto [i, t] : llvm::enumerate(
           llvm::cast<hugr_mlir::SumType>(pred.getType()).getTypes())) {
    auto tt = llvm::cast<TupleType>(t);
    auto v = rw.createOrFold<hugr_mlir::ReadVariantOp>(loc, tt, pred, i);
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

mlir::LogicalResult LowerHugrPass::initialize(MLIRContext* context) {
  RewritePatternSet ps(context);
  ps.add<LowerCfg, LowerHugrFuncToFunc, LowerOutput, LowerSwitch>(context);

  patterns =
      FrozenRewritePatternSet(std::move(ps), disabledPatterns, enabledPatterns);
  return success();
}

void LowerHugrPass::runOnOperation() {
  auto op = getOperation();
  auto context = &getContext();
  GreedyRewriteConfig cfg;
  cfg.useTopDownTraversal = true;
  cfg.enableRegionSimplification = true;
  bool changed = false;
  if (failed(applyPatternsAndFoldGreedily(op, patterns, cfg, &changed))) {
    emitError(op->getLoc(), "LowerHugrPass: Failed to apply patterns");
    return signalPassFailure();
  }

  if (hugrVerify) {
    ConversionTarget target(*context);
    target.addIllegalOp<
        hugr_mlir::FuncOp, hugr_mlir::CfgOp, hugr_mlir::DfgOp,
        hugr_mlir::TailLoopOp, hugr_mlir::ConditionalOp, hugr_mlir::OutputOp,
        hugr_mlir::SwitchOp>();
    if (failed(applyPartialConversion(op, target, {}))) {
      emitError(op->getLoc(), "Failed to convert all ops");
      return signalPassFailure();
    }
  }
  if (!changed) {
    markAllAnalysesPreserved();
  }
}
