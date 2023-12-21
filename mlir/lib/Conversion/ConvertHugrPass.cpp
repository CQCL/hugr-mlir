#include "hugr-mlir/Conversion/ConvertHugrPass.h"
#include "hugr-mlir/Conversion/TypeConverter.h"

#include "hugr-mlir/IR/HugrOps.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

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

mlir::LogicalResult ConvertHugrPass::initialize(MLIRContext* context) {
    type_converter = hugr_mlir::createTypeConverter();
    RewritePatternSet ps(context);

    // ps.add<ConvertHugrFuncToFuncConversionPattern>(*type_converter, context);
    ps.add<ConvertMakeTuple,ConvertUnpackTuple, ConvertTag, ConvertReadTag, ConvertReadVariant>(*type_converter, context);
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
