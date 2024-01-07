#include "hugr-mlir/Transforms/LowerHugrPass.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"

// #include "hugr-mlir/Conversion/ConvertHugrToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// struct LowerHugrTypeConverter : TypeConverter {
//     LowerHugrTypeConverter();
// };

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


struct LowerCall : OpRewritePattern<hugr_mlir::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::CallOp, PatternRewriter&) const override;
};

}  // namespace

// LowerHugrTypeConverter::LowerHugrTypeConverter() : TypeConverter() {
//     addConversion([](hugr_mlir::FunctionType t) -> std::optional<Type> {
//         return t.getFunctionType();
//     });

// }

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


mlir::LogicalResult LowerCall::matchAndRewrite(hugr_mlir::CallOp op, PatternRewriter & rw) const {
    if(auto callee = op.getCalleeAttrAttr()) {
        rw.replaceOpWithNewOp<func::CallOp>(op, callee.getRef(), op.getResultTypes(), op.getInputs());
    } // else if (auto callee = op.getCalleeValue()){
    //     rw.replaceOpWithNewOp<func::CallIndirectOp>(op, callee, op.getResultTypes(), op.getInputs());
    // }

    else { return failure(); }

    return success();
}

mlir::LogicalResult LowerHugrPass::initialize(MLIRContext* context) {
  RewritePatternSet ps(context);
  ps.add<LowerCfg, LowerHugrFuncToFunc, LowerOutput, LowerCall>(context);

  patterns =
      FrozenRewritePatternSet(std::move(ps), disabledPatterns, enabledPatterns);
  return success();
}

void LowerHugrPass::runOnOperation() {
  auto op = getOperation();
  auto context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<hugr_mlir::HugrDialect, func::FuncDialect,cf::ControlFlowDialect,scf::SCFDialect,index::IndexDialect>();
  target.addIllegalOp<
        hugr_mlir::FuncOp, hugr_mlir::CfgOp, hugr_mlir::DfgOp,
        hugr_mlir::TailLoopOp, hugr_mlir::ConditionalOp, hugr_mlir::OutputOp,
        hugr_mlir::SwitchOp>();

  target.addDynamicallyLegalOp<hugr_mlir::CallOp>([](hugr_mlir::CallOp op) -> bool {
      return !!op.getCalleeValue();
  });

  bool changed = false;
  if (failed(applyPartialConversion(op, target, patterns))) {
    emitError(op->getLoc(), "LowerHugrPass: Failed to apply patterns");
    return signalPassFailure();
  }
}
