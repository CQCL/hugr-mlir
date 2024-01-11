#include "hugr-mlir/Conversion/ConvertHugrPass.h"

#include "hugr-mlir/Conversion/TypeConverter.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

#define DEBUG_TYPE "convert-hugr-pass"

namespace hugr_mlir {
#define GEN_PASS_DEF_CONVERTHUGRPASS
#define GEN_PASS_DEF_CONVERTHUGRMODULEPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir

namespace {
using namespace mlir;

struct ConvertHugrPass : hugr_mlir::impl::ConvertHugrPassBase<ConvertHugrPass> {
  using ConvertHugrPassBase::ConvertHugrPassBase;
  LogicalResult initialize(MLIRContext*) override;
  void runOnOperation() override;

 private:
  FrozenRewritePatternSet conversion_patterns;
  FrozenRewritePatternSet lowering_patterns;
  std::shared_ptr<OneToNTypeConverter> type_converter;
};

struct ConvertMakeTuple : OneToNOpConversionPattern<hugr_mlir::MakeTupleOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::MakeTupleOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

struct ConvertUnpackTuple
    : OneToNOpConversionPattern<hugr_mlir::UnpackTupleOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::UnpackTupleOp, OpAdaptor,
      OneToNPatternRewriter&) const override;
};

struct ConvertTag : OneToNOpConversionPattern<hugr_mlir::TagOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::TagOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

struct ConvertReadTag : OneToNOpConversionPattern<hugr_mlir::ReadTagOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::ReadTagOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

struct ConvertReadVariant
    : OneToNOpConversionPattern<hugr_mlir::ReadVariantOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::ReadVariantOp, OpAdaptor,
      OneToNPatternRewriter&) const override;
};

struct LowerCfg : OpRewritePattern<hugr_mlir::CfgOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::CfgOp, PatternRewriter&) const override;
};

struct LowerDfg : OpRewritePattern<hugr_mlir::DfgOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::DfgOp, PatternRewriter&) const override;
};

struct LowerConditional : OpRewritePattern<hugr_mlir::ConditionalOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::ConditionalOp, PatternRewriter&) const override;
};

struct LowerEmptyReadClosure : OpRewritePattern<hugr_mlir::ReadClosureOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::ReadClosureOp, PatternRewriter&) const override;
};

struct LowerEmptyWriteClosure : OpRewritePattern<hugr_mlir::WriteClosureOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::WriteClosureOp, PatternRewriter&) const override;
};

// struct ConvertAllocClosure :
// OneToNOpConversionPattern<hugr_mlir::AllocClosureOp> {
//   using OneToNOpConversionPattern::OneToNOpConversionPattern;
//   LogicalResult matchAndRewrite(
//       hugr_mlir::AllocClosureOp, OpAdaptor, OneToNPatternRewriter&) const
//       override;
// };

struct ConvertFuncBlockArgs : OneToNOpConversionPattern<func::FuncOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult matchAndRewrite(
      func::FuncOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

struct ConvertConstant : OneToNOpConversionPattern<hugr_mlir::ConstantOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::ConstantOp, OpAdaptor, OneToNPatternRewriter&) const override;
  void initialize() { setHasBoundedRewriteRecursion(true); }
};

struct ConvertLoadConstant : OneToNOpConversionPattern<hugr_mlir::LoadConstantOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::LoadConstantOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

}  // namespace

mlir::LogicalResult ConvertMakeTuple::matchAndRewrite(
    hugr_mlir::MakeTupleOp op, OpAdaptor adaptor,
    OneToNPatternRewriter& rw) const {
  rw.replaceOp(op, adaptor.getFlatOperands(), adaptor.getResultMapping());
  return success();
}

mlir::LogicalResult ConvertUnpackTuple::matchAndRewrite(
    hugr_mlir::UnpackTupleOp op, OpAdaptor adaptor,
    OneToNPatternRewriter& rw) const {
  rw.replaceOp(op, adaptor.getFlatOperands(), adaptor.getResultMapping());
  return success();
}

mlir::LogicalResult ConvertTag::matchAndRewrite(
    hugr_mlir::TagOp op, OpAdaptor adaptor, OneToNPatternRewriter& rw) const {
  auto loc = op.getLoc();
  SmallVector<Value> results;
  auto st = op.getResult().getType();
  auto tag = op.getTag().getZExtValue();
  SmallVector<SmallVector<Type>> alt_types;
  for (auto i = 0; i < st.numAlts(); ++i) {
    if (failed(getTypeConverter()->convertTypes(
            st.getAltType(i), alt_types.emplace_back()))) {
      return failure();
    }
  }
  results.push_back(rw.createOrFold<index::ConstantOp>(loc, op.getTagAttr()));
  for (auto i = 0; i < st.numAlts(); ++i) {
    if (i == tag) {
      assert(alt_types[i] == adaptor.getInput().getTypes() && "must");
      llvm::copy(adaptor.getInput(), std::back_inserter(results));
    } else {
      llvm::transform(
          alt_types[i], std::back_inserter(results), [&rw, loc](Type t) {
            return rw.createOrFold<ub::PoisonOp>(
                loc, t, rw.getAttr<ub::PoisonAttr>());
          });
    }
  }
  rw.replaceOp(op, results, adaptor.getResultMapping());
  return success();
}

mlir::LogicalResult ConvertReadTag::matchAndRewrite(
    hugr_mlir::ReadTagOp op, OpAdaptor adaptor,
    OneToNPatternRewriter& rw) const {
  rw.replaceOp(op, adaptor.getInput()[0]);
  return success();
}

mlir::LogicalResult ConvertReadVariant::matchAndRewrite(
    hugr_mlir::ReadVariantOp op, OpAdaptor adaptor,
    OneToNPatternRewriter& rw) const {
  auto st = op.getInput().getType();
  auto loc = op.getLoc();
  if (st.numAlts() == 0) {
    return failure();
  }
  OneToNTypeMapping mapping(st.getTypes());
  if (failed(
          getTypeConverter()->convertSignatureArgs(st.getTypes(), mapping))) {
    return failure();
  }
  assert(
      adaptor.getInput().size() == mapping.getConvertedTypes().size() + 1 &&
      "must");

  // auto tag = adaptor.getInput().front();
  rw.replaceOp(
      op,
      mapping.getConvertedValues(
          adaptor.getInput().drop_front(), op.getTag().getZExtValue()),
      adaptor.getResultMapping());

  return success();
}

mlir::LogicalResult ConvertFuncBlockArgs::matchAndRewrite(
    func::FuncOp op, OpAdaptor adaptor, OneToNPatternRewriter& rw) const {
  auto tc = getTypeConverter<OneToNTypeConverter>();
  if (!tc) {
    return rw.notifyMatchFailure(op, "no type converter");
  }

  struct WorkItem {
    WorkItem(Block& b, OneToNTypeMapping mapping_)
        : target(b), mapping(mapping_) {
      llvm::copy(b.getPredecessors(), std::back_inserter(preds));
    }
    Block& target;
    SmallVector<Block*> preds;
    OneToNTypeMapping mapping;
    std::tuple<BranchOpInterface, unsigned, SuccessorOperands>
    getPredecessorBranchOpAndSuccessorOperands(Block* pred) const {
      assert(llvm::is_contained(preds, pred) && "precondition");
      auto boi = llvm::cast<BranchOpInterface>(pred->getTerminator());
      auto succ_i_ = llvm::find(boi->getSuccessors(), &target);
      assert(succ_i_ != boi->getSuccessors().end() && "must");
      auto succ_i = succ_i_ - boi->getSuccessors().begin();
      return {boi, succ_i, boi.getSuccessorOperands(succ_i)};
    }

    bool allPredsAreBranchOpInterfaces() const {
      return llvm::all_of(preds, [this](auto p) {
        if (auto boi = llvm::dyn_cast<BranchOpInterface>(p->getTerminator())) {
          auto [_, __, succ_ops] =
              getPredecessorBranchOpAndSuccessorOperands(p);
          auto prod = succ_ops.getProducedOperandCount();
          return mapping.getConvertedTypes().take_front(prod) ==
                 mapping.getOriginalTypes().take_front(prod);
        }
        return false;
      });
    }
  };
  auto mk_work_item = [&tc](Block& b) -> FailureOr<WorkItem> {
    if (b.isEntryBlock() || tc->isLegal(b.getArgumentTypes())) {
      return failure();
    }
    WorkItem wi(WorkItem(b, OneToNTypeMapping(b.getArgumentTypes())));
    if (!wi.allPredsAreBranchOpInterfaces() ||
        failed(tc->convertSignatureArgs(b.getArgumentTypes(), wi.mapping))) {
      return failure();
    }
    return std::move(wi);
  };
  std::vector<WorkItem> worklist;
  for (auto& r : op->getRegions()) {
    for (auto& b : r.getBlocks()) {
      auto mb_wi = mk_work_item(b);
      if (failed(mb_wi)) {
        continue;
      }
      worklist.push_back(std::move(*mb_wi));
    }
  }
  if (worklist.empty()) {
    return rw.notifyMatchFailure(op, "no work to do");
  }

  std::vector<std::tuple<Block*, OneToNTypeMapping>> signature_conversions;
  LLVM_DEBUG(llvm::dbgs() << "ConvertBlockArgs:" << worklist.size() << "\n");
  rw.startRootUpdate(op);
  for (auto const& wi : worklist) {
    assert(wi.mapping.hasNonIdentityConversion() && "must");

    for (auto pred : wi.preds) {
      auto [boi, succ_i, succ_ops] =
          wi.getPredecessorBranchOpAndSuccessorOperands(pred);
      SmallVector<Value> new_forwarded_operands;
      rw.setInsertionPoint(boi);
      for (auto [i, fo] : llvm::enumerate(succ_ops.getForwardedOperands())) {
        if (auto new_vals = tc->materializeTargetConversion(
                rw, pred->getTerminator()->getLoc(),
                wi.mapping.getConvertedTypes(i), fo)) {
          llvm::copy(*new_vals, std::back_inserter(new_forwarded_operands));
        } else {
          assert(false && "failed materialization");
        }
      }
      assert(
          TypeRange(new_forwarded_operands) !=
              succ_ops.getForwardedOperands().getTypes() &&
          "non identity");
      rw.startRootUpdate(boi);
      succ_ops.getMutableForwardedOperands().assign(new_forwarded_operands);
      rw.finalizeRootUpdate(boi);
    }
    signature_conversions.emplace_back(&wi.target, wi.mapping);
  }
  if (signature_conversions.size()) {
    for (auto& [target, sig] : signature_conversions) {
      rw.applySignatureConversion(target, sig);
    }
    rw.finalizeRootUpdate(op);
    return success();
  }
  rw.cancelRootUpdate(op);
  return rw.notifyMatchFailure(op, "no merges");
}

LogicalResult ConvertConstant::matchAndRewrite(
    hugr_mlir::ConstantOp op, OpAdaptor adaptor,
    OneToNPatternRewriter& rw) const {
  return llvm::TypeSwitch<Attribute, LogicalResult>(op.getValue())
      .Case([&](IntegerAttr a) {
        return llvm::TypeSwitch<Type, LogicalResult>(a.getType())
          .Case([&](IndexType t) {
            rw.replaceOpWithNewOp<index::ConstantOp>(op, t, a);
            return success();
          }).Case([&](IntegerType t) {
            rw.replaceOpWithNewOp<arith::ConstantOp>(op, t, a);
            return success();
          }).Default([](auto) { return failure(); } );
      }).Case([&](FloatAttr a) {
          rw.replaceOpWithNewOp<arith::ConstantOp>(op, a.getType(), a);
          return success();
      }).Case([&](hugr_mlir::TupleAttr ta) {
        SmallVector<Value> results;
        for (auto a : ta.getValues()) {
          auto c_op = a.getDialect().materializeConstant(
              rw, a, a.getType(), op.getLoc());
          assert(
              c_op->getNumResults() == 1 && "contract of materializeConstant");

          SmallVector<Type> converted_tys;
          if (failed(getTypeConverter()->convertType(
                  a.getType(), converted_tys))) {
            assert(false && "awk");
          }

          auto mb_results =
              getTypeConverter<OneToNTypeConverter>()
                  ->materializeTargetConversion(
                      rw, op.getLoc(), converted_tys, c_op->getResult(0));
          if (!mb_results) {
            assert(false && "awk2");
          }
          llvm::copy(*mb_results, std::back_inserter(results));
        }
        rw.replaceOp(op, results, adaptor.getResultMapping());
        return success();
      })
      .Case([&](hugr_mlir::SumAttr sa) {
        SmallVector<Value> results;
        auto tag_c_op = getContext()
                            ->getLoadedDialect<index::IndexDialect>()
                            ->materializeConstant(
                                rw, sa.getTagAttr(), sa.getTagAttr().getType(),
                                op.getLoc());
        assert(
            tag_c_op && tag_c_op->getNumResults() == 1 &&
            "contract of materializeConstant");
        results.push_back(tag_c_op->getResult(0));

        for (auto [i, t] : llvm::enumerate(sa.getSumType().getTypes())) {
          SmallVector<Type> converted_tys;
          if (failed(getTypeConverter()->convertType(t, converted_tys))) {
            assert(false && "awk");
          }
          if (i == sa.getTag()) {
            auto c_op = sa.getValue().getDialect().materializeConstant(
                rw, sa.getValue(), t, op.getLoc());
            LLVM_DEBUG(llvm::dbgs() << sa.getValue() << "," << t << ", " << sa.getValue().getDialect().getNamespace() << "\n");
            assert(
              c_op && c_op->getNumResults() == 1 &&
                "contract of materializeConstant");
            auto mb_results =
                getTypeConverter<OneToNTypeConverter>()
                    ->materializeTargetConversion(
                        rw, op.getLoc(), converted_tys, c_op->getResult(0));
            if (!mb_results) {
              assert(false && "awk2");
            }
            llvm::copy(*mb_results, std::back_inserter(results));
          } else {
            llvm::transform(
                converted_tys, std::back_inserter(results), [&](auto ct) {
                  return rw.createOrFold<ub::PoisonOp>(
                      op.getLoc(), ct, rw.getAttr<ub::PoisonAttr>());
                });
          }
        }
        rw.replaceOp(op, results, adaptor.getResultMapping());
        return success();
      })
      .Default([](auto) { return failure(); });
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

mlir::LogicalResult LowerDfg::matchAndRewrite(
    hugr_mlir::DfgOp op, PatternRewriter& rw) const {
  hugr_mlir::OutputOp dfg_output;
  if (op.getBody().empty() ||
      !(dfg_output = llvm::dyn_cast<hugr_mlir::OutputOp>(
            op.getBody().front().getTerminator()))) {
    return failure();
  }
  auto entry_block = op->getBlock();
  auto split_block = rw.splitBlock(entry_block, Block::iterator{op});
  rw.inlineBlockBefore(
      &op.getBody().front(), entry_block, entry_block->end(), op.getInputs());
  rw.inlineBlockBefore(split_block, entry_block, entry_block->end());
  rw.replaceOp(op, dfg_output.getOutputs());
  rw.eraseOp(dfg_output);
  return success();
}

mlir::LogicalResult LowerConditional::matchAndRewrite(
    hugr_mlir::ConditionalOp op, PatternRewriter& rw) const {
  auto loc = op.getLoc();
  auto pred = op.getPredicate();
  auto pred_ty = llvm::cast<hugr_mlir::SumType>(pred.getType());
  if(pred_ty.numAlts() == 0) { return failure(); }
  auto orig_block = op->getBlock();

  auto tail_block = rw.splitBlock(op->getBlock(), Block::iterator{op});
  SmallVector<Type> exit_block_tys;
  SmallVector<Location> exit_block_locs;
  for(auto x: op.getResults()) {
    exit_block_tys.push_back(x.getType());
    exit_block_locs.push_back(x.getLoc());
  }
  auto exit_block = rw.createBlock(tail_block, exit_block_tys, exit_block_locs);
  rw.mergeBlocks(tail_block, exit_block);

  SmallVector<int32_t> case_values;
  for (auto i = 0; i < pred_ty.numAlts(); ++i) {
    case_values.push_back(i);
  }
  SmallVector<Block*> case_destinations;
  for(auto [i, r]: llvm::enumerate(op.getCases())) {
    auto case_block = &r.front();
    auto output = llvm::cast<hugr_mlir::OutputOp>(case_block->getTerminator());
    rw.inlineRegionBefore(r, exit_block);
    rw.setInsertionPoint(output);
    rw.replaceOpWithNewOp<cf::BranchOp>(output, exit_block, output.getOutputs());
    auto entry_block = rw.createBlock(case_block);
    rw.setInsertionPointToStart(entry_block);
    auto tup_val = rw.createOrFold<hugr_mlir::ReadVariantOp>(loc, pred, i);
    SmallVector<Value> replacements;
    rw.createOrFold<hugr_mlir::UnpackTupleOp>(replacements, loc, tup_val);
    llvm::copy(op.getOtherInputs(), std::back_inserter(replacements));
    rw.mergeBlocks(case_block, entry_block, replacements);
    case_destinations.push_back(entry_block);
  }

  rw.setInsertionPointToEnd(orig_block);
  auto tag = rw.createOrFold<index::CastUOp>(
      loc, rw.getI32Type(), rw.createOrFold<hugr_mlir::ReadTagOp>(loc, pred));
  rw.create<cf::SwitchOp>(loc, tag, case_destinations[0], ValueRange{}, ArrayRef(case_values).drop_front(), BlockRange{case_destinations}.drop_front(), SmallVector(pred_ty.numAlts() - 1, ValueRange{}));

  rw.replaceOp(op, exit_block->getArguments());
  return success();
}

mlir::LogicalResult LowerEmptyReadClosure::matchAndRewrite(
    hugr_mlir::ReadClosureOp op, PatternRewriter& rw) const {
  // TODO this should be done by canonicalisation
  if (!op.getResults().empty()) {
    return failure();
  }
  rw.eraseOp(op);
  return success();
}

mlir::LogicalResult LowerEmptyWriteClosure::matchAndRewrite(
    hugr_mlir::WriteClosureOp op, PatternRewriter& rw) const {
  // TODO this should be done by canonicalisation
  if (!op.getInputs().empty()) {
    return failure();
  }
  rw.eraseOp(op);
  return success();
}

mlir::LogicalResult ConvertLoadConstant::matchAndRewrite(hugr_mlir::LoadConstantOp op, OpAdaptor adaptor, OneToNPatternRewriter &rw) const {
  auto const_ref = op.getConstRef();
  // this is inefficient, we should pass a map of const ops into the pattern
  auto referee0 = SymbolTable::lookupNearestSymbolFrom(op, const_ref.getRef());
  // this can fail when enclosing cfgs/dfgs have yet to be lowered
  if(!referee0) {
    return rw.notifyMatchFailure(op, [&](auto& d) {
      d << "Unknown symbol: " << const_ref.getRef();
    });
  }
  assert(referee0 && "by verification");
  auto referee = llvm::cast<hugr_mlir::ConstOp>(referee0);

  rw.replaceOpWithNewOp<hugr_mlir::ConstantOp>(op, *referee.getValue());
  return success();
}

mlir::LogicalResult ConvertHugrPass::initialize(MLIRContext* context) {
  type_converter = hugr_mlir::createTypeConverter();
  {
    RewritePatternSet ps(context);
    // TODO lower TailLoopOp here
    ps.add<LowerCfg, LowerDfg, LowerConditional, LowerEmptyReadClosure, LowerEmptyWriteClosure>(
        context);
    lowering_patterns = FrozenRewritePatternSet(
        std::move(ps), disabledPatterns, enabledPatterns);
  }
  {
    RewritePatternSet ps(context);

    // ps.add<ConvertHugrFuncToFuncConversionPattern>(*type_converter, context);
    ps.add<
        ConvertConstant, ConvertMakeTuple, ConvertUnpackTuple, ConvertTag,
        ConvertReadTag, ConvertReadVariant, ConvertLoadConstant>(*type_converter, context);
    ps.add<ConvertFuncBlockArgs>(*type_converter, context);
    populateFuncTypeConversionPatterns(*type_converter, ps);
    scf::populateSCFStructuralOneToNTypeConversions(*type_converter, ps);

    conversion_patterns = FrozenRewritePatternSet(
        std::move(ps), disabledPatterns, enabledPatterns);
  }
  return success();
};

void ConvertHugrPass::runOnOperation() {
  auto op = getOperation();
  auto context = &getContext();

  {
    GreedyRewriteConfig cfg;
    cfg.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(op, lowering_patterns, cfg))) {
      emitError(
          op->getLoc(), "LowerHugrPass: Failed to apply lowering patterns");
      return signalPassFailure();
    }
  }

  if (failed(applyPartialOneToNConversion(
          op, *type_converter,
          conversion_patterns))) {
    emitError(
        op->getLoc(),
        "ConvertHugrPass: failure to applyPartialOneToNConversion");
    return signalPassFailure();
  }

  if (hugrVerify) {
    std::optional<InFlightDiagnostic> mb_ifd;
    auto ifd = [&]() -> InFlightDiagnostic& {
      if(!mb_ifd) {
        mb_ifd.emplace(std::move(emitError(getOperation()->getLoc(), "ConvertHugrPass: Failed to convert ops:")));
      }
      return *mb_ifd;
    };

    ConversionTarget target(*context);
    target.addIllegalDialect<hugr_mlir::HugrDialect>();
    target.addLegalOp<hugr_mlir::ExtensionOp, hugr_mlir::ConstOp>();
    getOperation()->walk([&](Operation* op) {
      if(target.isIllegal(op)) {
        ifd().attachNote(op->getLoc()) << ":" << op->getName();
      }
    });
    if(mb_ifd) {return signalPassFailure();}
  }
}
