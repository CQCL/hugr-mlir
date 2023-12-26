#include "hugr-mlir/Conversion/ConvertHugrPass.h"
#include "hugr-mlir/Conversion/TypeConverter.h"

#include "hugr-mlir/IR/HugrOps.h"
#include "hugr-mlir/IR/HugrDialect.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "convert-hugr-pass"

namespace hugr_mlir {
#define GEN_PASS_DEF_CONVERTHUGRPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir

namespace {
using namespace mlir;

struct ConvertHugrPass : hugr_mlir::impl::ConvertHugrPassBase<ConvertHugrPass> {
  using  ConvertHugrPassBase::ConvertHugrPassBase;
  LogicalResult initialize(MLIRContext*) override;
  void runOnOperation() override;

private:
  FrozenRewritePatternSet patterns;
  std::shared_ptr<TypeConverter> type_converter;
};



// struct ConvertSwitchConversionPattern : OneToNConversionPattern<hugr_mlir::SwitchOp> {
//     using OpConversionPattern::OpConversionPattern;
//     LogicalResult matchAndRewrite(hugr_mlir::SwitchOp, hugr_mlir::SwitchOpAdaptor, ConversionPatternRewriter&) const override;
// };

struct ConvertMakeTuple : OneToNOpConversionPattern<hugr_mlir::MakeTupleOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;
    LogicalResult matchAndRewrite(hugr_mlir::MakeTupleOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

struct ConvertUnpackTuple : OneToNOpConversionPattern<hugr_mlir::UnpackTupleOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;
    LogicalResult matchAndRewrite(hugr_mlir::UnpackTupleOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

struct ConvertTag : OneToNOpConversionPattern<hugr_mlir::TagOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;
    LogicalResult matchAndRewrite(hugr_mlir::TagOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

struct ConvertReadTag : OneToNOpConversionPattern<hugr_mlir::ReadTagOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;
    LogicalResult matchAndRewrite(hugr_mlir::ReadTagOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

struct ConvertReadVariant : OneToNOpConversionPattern<hugr_mlir::ReadVariantOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;
    LogicalResult matchAndRewrite(hugr_mlir::ReadVariantOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

struct ConvertBlockArgs : OneToNConversionPattern {
    ConvertBlockArgs(TypeConverter &typeConverter, MLIRContext *context,
                            PatternBenefit benefit = 1,
                            ArrayRef<StringRef> generatedNames = {})
        : OneToNConversionPattern(typeConverter, Pattern::MatchAnyOpTypeTag(), benefit, context, generatedNames) {}
    LogicalResult matchAndRewrite(Operation *op,
                                        OneToNPatternRewriter &rewriter,
                                        const OneToNTypeMapping &operandMapping,
                                        const OneToNTypeMapping &resultMapping,
                                        ValueRange convertedOperands) const override;
    void initialize() {
        this->setHasBoundedRewriteRecursion();
    }
};

struct ConvertConstant : OneToNOpConversionPattern<hugr_mlir::ConstantOp> {
    using OneToNOpConversionPattern::OneToNOpConversionPattern;
    LogicalResult matchAndRewrite(hugr_mlir::ConstantOp, OpAdaptor, OneToNPatternRewriter&) const override;
};

}


mlir::LogicalResult ConvertMakeTuple::matchAndRewrite(hugr_mlir::MakeTupleOp op, OpAdaptor adaptor, OneToNPatternRewriter& rw) const {
    rw.replaceOp(op, adaptor.getFlatOperands(), adaptor.getResultMapping());
    return success();
}

mlir::LogicalResult ConvertUnpackTuple::matchAndRewrite(hugr_mlir::UnpackTupleOp op, OpAdaptor adaptor, OneToNPatternRewriter& rw) const {
    rw.replaceOp(op, adaptor.getFlatOperands(), adaptor.getResultMapping());
    return success();
}

mlir::LogicalResult ConvertTag::matchAndRewrite(hugr_mlir::TagOp op, OpAdaptor adaptor, OneToNPatternRewriter& rw) const {
    auto loc = op.getLoc();
    SmallVector<Value> results;
    auto st = op.getResult().getType();
    auto tag = op.getTag().getZExtValue();
    SmallVector<SmallVector<Type>> alt_types;
    for(auto i = 0; i < st.numAlts(); ++i) {
        if(failed(getTypeConverter()->convertTypes(st.getAltType(i), alt_types.emplace_back()))) {
            return failure();
        }
    }
    results.push_back(rw.createOrFold<index::ConstantOp>(loc, op.getTagAttr()));
    for(auto i = 0; i < st.numAlts(); ++i) {
        if(i == tag) {
            assert(alt_types[i] == adaptor.getInput().getTypes() && "must");
            llvm::copy(adaptor.getInput(), std::back_inserter(results));
        } else {
            llvm::transform(alt_types[i], std::back_inserter(results), [&rw,loc](Type t) {
                return rw.createOrFold<ub::PoisonOp>(loc, t, rw.getAttr<ub::PoisonAttr>());
            });
        }
    }
    rw.replaceOp(op, results, adaptor.getResultMapping());
    return success();
}

mlir::LogicalResult ConvertReadTag::matchAndRewrite(hugr_mlir::ReadTagOp op, OpAdaptor adaptor, OneToNPatternRewriter& rw) const {
    rw.replaceOp(op, adaptor.getInput()[0]);
    return success();
}

mlir::LogicalResult ConvertReadVariant::matchAndRewrite(hugr_mlir::ReadVariantOp op, OpAdaptor adaptor, OneToNPatternRewriter& rw) const {
    auto st = op.getInput().getType();
    auto loc = op.getLoc();
    if(st.numAlts() == 0) { return failure(); }
    OneToNTypeMapping mapping(st.getTypes());
    if(failed(getTypeConverter()->convertSignatureArgs(st.getTypes(), mapping))) {
        return failure();
    }
    assert(adaptor.getInput().size() == mapping.getConvertedTypes().size() + 1 && "must");

    // auto tag = adaptor.getInput().front();
    rw.replaceOp(op, mapping.getConvertedValues(adaptor.getInput().drop_front(), op.getTag().getZExtValue()), adaptor.getResultMapping());

    return success();
}

mlir::LogicalResult ConvertBlockArgs::matchAndRewrite(Operation *op,
                                    OneToNPatternRewriter &rw,
                                    const OneToNTypeMapping &operandMapping,
                                    const OneToNTypeMapping &resultMapping,
                              ValueRange convertedOperands) const {
    auto tc = getTypeConverter<OneToNTypeConverter>();
    if(!tc) { return failure(); }

    struct WorkItem {
        WorkItem(Block& b, OneToNTypeMapping mapping_) : target(b), mapping(mapping_) {
            llvm::copy(b.getPredecessors(), std::back_inserter(preds));
        }
        Block& target;
        SmallVector<Block*> preds;
        OneToNTypeMapping mapping;
        std::tuple<BranchOpInterface, unsigned, SuccessorOperands> getPredecessorBranchOpAndSuccessorOperands(Block* pred) const {
            assert(llvm::is_contained(preds, pred) && "precondition");
            auto boi = llvm::cast<BranchOpInterface>(pred->getTerminator());
            auto succ_i_ = llvm::find(boi->getSuccessors(), &target);
            assert(succ_i_ != boi->getSuccessors().end() && "must");
            auto succ_i = succ_i_ - boi->getSuccessors().begin();
            return {boi, succ_i, boi.getSuccessorOperands(succ_i)};
        }

        bool allPredsAreBranchOpInterfaces() const {
            return llvm::all_of(preds, [this](auto p) {
                if(auto boi = llvm::dyn_cast<BranchOpInterface>(p->getTerminator())) {
                    auto [_, __, succ_ops] = getPredecessorBranchOpAndSuccessorOperands(p);
                    auto prod = succ_ops.getProducedOperandCount();
                    return mapping.getConvertedTypes().take_front(prod) == mapping.getOriginalTypes().take_front(prod);
                }
                return false;
            });
        }
    };
    auto mk_work_item = [&tc](Block& b) -> FailureOr<WorkItem> {
      auto mapping = OneToNTypeMapping(b.getArgumentTypes());
      if(failed(tc->convertSignatureArgs(b.getArgumentTypes(), mapping))) {
          return failure();
      }
      WorkItem wi(WorkItem(b, std::move(mapping)));
      if(!wi.allPredsAreBranchOpInterfaces()) { return failure(); }
      return std::move(wi);
    };
    std::vector<WorkItem> worklist;
    for(auto& r: op->getRegions()) {
        for(auto& b: r.getBlocks()) {
            if(b.isEntryBlock()) { continue; }
            if(tc->isLegal(b.getArgumentTypes())) { continue; }
            auto mb_wi = mk_work_item(b);
            if(failed(mb_wi) || !mb_wi->allPredsAreBranchOpInterfaces()) { continue; }
            worklist.push_back(std::move(*mb_wi));
        }
    }
    if(worklist.empty()) { return failure(); }

    SmallVector<std::tuple<Block*,Block*,SmallVector<Value>>> merges;
    LLVM_DEBUG(llvm::dbgs() << "ConvertBlockArgs:" << worklist.size() << "\n");
    rw.startRootUpdate(op);
    for(auto const& wi: worklist) {
        assert(wi.mapping.hasNonIdentityConversion() && "must");
        SmallVector<Type> new_types{wi.mapping.getConvertedTypes()};
        SmallVector<Location> new_locs;
        wi.mapping.convertLocations(wi.target.getArguments(), new_locs);
        auto new_block = rw.createBlock(&wi.target, new_types, new_locs);

        for(auto pred: wi.preds) {
            auto [boi, succ_i, succ_ops] = wi.getPredecessorBranchOpAndSuccessorOperands(pred);
            SmallVector<Value> new_forwarded_operands;
            rw.setInsertionPoint(boi);
            for(auto [i, fo]: llvm::enumerate(succ_ops.getForwardedOperands())) {
                if(auto new_vals = tc->materializeTargetConversion(rw, pred->getTerminator()->getLoc(), wi.mapping.getConvertedTypes(i), fo)) {
                    llvm::copy(*new_vals, std::back_inserter(new_forwarded_operands));
                } else {
                    assert(false && "failed materialization");
                }
            }
            assert(TypeRange(new_forwarded_operands) != succ_ops.getForwardedOperands().getTypes() && "non identity");
            rw.startRootUpdate(boi);
            succ_ops.getMutableForwardedOperands().assign(new_forwarded_operands);
            boi->setSuccessor(new_block, succ_i);
            rw.finalizeRootUpdate(boi);
        }
        rw.setInsertionPointToStart(new_block);
        SmallVector<Value> source_args;
        llvm::transform(wi.target.getArguments(), std::back_inserter(source_args), [&](BlockArgument a) {
            return tc->materializeSourceConversion(rw, a.getLoc(), a.getType(), wi.mapping.getConvertedValues(new_block->getArguments(), a.getArgNumber()));
        });
        merges.push_back(std::make_tuple(&wi.target, new_block, std::move(source_args)));
    }
    if(merges.size()) {
        for(auto& [source, dest, values] : merges) {
            rw.mergeBlocks(source, dest, values);
        }
        rw.finalizeRootUpdate(op);
        return success();
    }
    rw.cancelRootUpdate(op);
    return failure();
}

LogicalResult ConvertConstant::matchAndRewrite(hugr_mlir::ConstantOp op, OpAdaptor adaptor, OneToNPatternRewriter& rw) const {
    return llvm::TypeSwitch<Attribute, LogicalResult>(op.getValue())
        .Case([&](hugr_mlir::TupleAttr ta) {
            SmallVector<Value> results;
            for(auto a: ta.getValues()) {
                auto c_op = a.getDialect().materializeConstant(rw, a, a.getType(), op.getLoc());
                assert(c_op->getNumResults() == 1 && "contract of materializeConstant");

                SmallVector<Type> converted_tys;
                if(failed(getTypeConverter()->convertType(a.getType(), converted_tys))) {
                    assert(false && "awk");
                }

                auto mb_results = getTypeConverter<OneToNTypeConverter>()->materializeTargetConversion(rw, op.getLoc(), converted_tys, c_op->getResult(0));
                if(!mb_results) {
                    assert(false && "awk2");
                }
                llvm::copy(*mb_results, std::back_inserter(results));
            }
            rw.replaceOp(op, results, adaptor.getResultMapping());
            return success();
        }).Case([&](hugr_mlir::SumAttr sa) {
            SmallVector<Value> results;
            auto tag_c_op = getContext()->getLoadedDialect<index::IndexDialect>()->materializeConstant(rw, sa.getTagAttr(), sa.getTagAttr().getType(), op.getLoc());
            assert(tag_c_op && tag_c_op->getNumResults() == 1 && "contract of materializeConstant");
            results.push_back(tag_c_op->getResult(0));

            for(auto [i, t]: llvm::enumerate(sa.getSumType().getTypes())) {
                SmallVector<Type> converted_tys;
                if(failed(getTypeConverter()->convertType(t, converted_tys))) {
                    assert(false && "awk");
                }
                if(i == sa.getTag()) {
                    auto c_op = sa.getValue().getDialect().materializeConstant(rw, sa.getValue(), t, op.getLoc());
                    assert(c_op->getNumResults() == 1 && "contract of materializeConstant");
                    auto mb_results = getTypeConverter<OneToNTypeConverter>()->materializeTargetConversion(rw, op.getLoc(), converted_tys, c_op->getResult(0));
                    if(!mb_results) {
                        assert(false && "awk2");
                    }
                    llvm::copy(*mb_results, std::back_inserter(results));
                } else {
                    llvm::transform(converted_tys, std::back_inserter(results), [&](auto ct) {
                        return rw.createOrFold<ub::PoisonOp>(op.getLoc(), ct, rw.getAttr<ub::PoisonAttr>());
                    });
                }
            }
            rw.replaceOp(op, results, adaptor.getResultMapping());
            return success();
        }).Default([](auto) { return failure();});
}

mlir::LogicalResult ConvertHugrPass::initialize(MLIRContext* context) {
    type_converter = hugr_mlir::createTypeConverter();
    RewritePatternSet ps(context);

    // ps.add<ConvertHugrFuncToFuncConversionPattern>(*type_converter, context);
    ps.add<ConvertConstant,ConvertMakeTuple,ConvertUnpackTuple, ConvertTag, ConvertReadTag, ConvertReadVariant>(*type_converter, context);
    ps.add<ConvertBlockArgs>(*type_converter, context);
    populateFuncTypeConversionPatterns(*type_converter, ps);
    scf::populateSCFStructuralOneToNTypeConversions(*type_converter, ps);

    patterns = FrozenRewritePatternSet(std::move(ps), disabledPatterns, enabledPatterns);
    return success();
};

void ConvertHugrPass::runOnOperation() {
    auto op = getOperation();
    auto context = &getContext();
    // ConversionTarget target(*context);

    // target.addLegalDialect<cf::ControlFlowDialect,func::FuncDialect,arith::ArithDialect,scf::SCFDialect>();
    // target.addIllegalOp<hugr_mlir::FuncOp,hugr_mlir::CfgOp>();

    // target.addLegalOp<UnrealizedConversionCastOp>();
    // target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    //     return type_converter->isSignatureLegal(op.getFunctionType());
    // });
    // target.addDynamicallyLegalOp<hugr_mlir::OutputOp>([](hugr_mlir::OutputOp op) {
    //     auto parent = op->getParentOp();
    //     if(!parent) { return true; }
    //     if(llvm::isa<func::FuncOp>(parent)) {
    //         return false;
    //     }
    //     return true;
    // });

    if(failed(applyPartialOneToNConversion(op, static_cast<OneToNTypeConverter &>(*type_converter), patterns))) {
        emitError(op->getLoc(), "ConvertHugrPass: failure to applyPartialOneToNConversion");
        return signalPassFailure();
    }

    // GreedyRewriteConfig cfg;
    // cfg.useTopDownTraversal = true;
    // bool changed = false;
    // if(failed(applyPatternsAndFoldGreedily(op, patterns, cfg, &changed))) {
    //     emitError(op->getLoc(), "LowerHugrPass: Failed to apply patterns");
    //     return signalPassFailure();
    // }
    // if(!changed) {
    //     markAllAnalysesPreserved();
    // }

}
