#include "hugr-mlir/Conversion/ConvertHugrExtQuantumPass.h"

#include "hugr-mlir/Conversion/Utils.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"
#include "hugr-mlir/IR/HugrTypes.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/Sequence.h"

#define DEBUG_TYPE "convert-hugr-ext-quantum"

namespace hugr_mlir {
#define GEN_PASS_DEF_CONVERTHUGREXTQUANTUMPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir


namespace {
};



namespace {
using namespace mlir;

struct QIR_FuncData {
    // QIR_FuncData(QIR_FuncData const&) = default;
    // QIR_FuncData& operator=(QIR_FuncData const&) = default;
    StringRef symbol{};
    FunctionType type{nullptr};
};

enum class QIR_FuncIndex : unsigned {
  H,
  Measure,
  QAlloc,
  QFree,
  QIR_Result0,
  QIR_Result1,
  QIR_ResultEq,
  _END
};


struct QIR_FuncDataFactory {
    QIR_FuncDataFactory(MLIRContext* context_) : context(context_) {}

    QIR_FuncData get(QIR_FuncIndex) const;
    SmallVector<QIR_FuncData> all() const;
protected:
    LLVM::LLVMPointerType getPointerType() const { return LLVM::LLVMPointerType::get(context); };
    MLIRContext * const context;
};

struct QuantumTypeConverter : TypeConverter {

    QuantumTypeConverter() : TypeConverter() {
        addConversion([](Type t) { return t; });
        addConversion([this](hugr_mlir::OpaqueType t) -> std::optional<Type> {
            if(is_qubit(t)) {
                return LLVM::LLVMPointerType::get(t.getContext());
            }
            return std::nullopt;
        });
        addArgumentMaterialization([](OpBuilder& rw, Type t, ValueRange vs, Location loc) -> std::optional<Value> {
            if(vs.size() != 1 || !is_qubit(vs[0].getType())) { return std::nullopt; }
            auto c_op = rw.create<UnrealizedConversionCastOp>(loc, t, vs);
            return c_op.getResult(0);
        });
        addTargetMaterialization([](OpBuilder& rw, Type t, ValueRange vs, Location loc) -> std::optional<Value> {
            if(vs.size() != 1 || !is_qubit(vs[0].getType())) { return std::nullopt; }
            auto c_op = rw.create<UnrealizedConversionCastOp>(loc, t, vs);
            return c_op.getResult(0);
        });
        addSourceMaterialization([](OpBuilder& rw, Type t, ValueRange vs, Location loc) -> std::optional<Value> {
            if(vs.size() != 1 || !is_qubit(t)) { return std::nullopt; }
            auto c_op = rw.create<UnrealizedConversionCastOp>(loc, t, vs);
            return c_op.getResult(0);
        });
    }
private:
    static bool is_qubit(Type t)  {
        if(auto o = llvm::dyn_cast<hugr_mlir::OpaqueType>(t)) {
            if(o.getExtension().getName() == "prelude" && o.getTypeArgs().empty() && o.getName() == "qubit") {
                return true;
            }
        }
        return false;
    };


};

struct ConvertHugrExtQuantumPass : hugr_mlir::impl::ConvertHugrExtQuantumPassBase<ConvertHugrExtQuantumPass> {
    using ConvertHugrExtQuantumPassBase::ConvertHugrExtQuantumPassBase;
    LogicalResult initialize(MLIRContext*) override;
    void runOnOperation() override;
private:
    std::shared_ptr<QuantumTypeConverter> type_converter;
    FrozenRewritePatternSet patterns;
};

struct QuantumPatternBase : hugr_mlir::HugrExtensionOpConversionPattern {
    template<typename ...Args>
    QuantumPatternBase(mlir::StringRef opname, int num_args, int num_results, Args& ...args) :
        hugr_mlir::HugrExtensionOpConversionPattern("quantum.tket2", opname, num_args, num_results, std::forward<Args>(args)...),
        funcdata(getContext()) {}
    // LogicalResult replace(hugr_mlir::ExtensionOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
protected:
    QIR_FuncDataFactory const funcdata;
};

struct H_Pattern : QuantumPatternBase {
    template<typename ...Args>
    H_Pattern(Args&& ...args) : QuantumPatternBase("H", 1, 1, std::forward<Args>(args)...) {}
    LogicalResult replace(hugr_mlir::ExtensionOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

struct Measure_Pattern : QuantumPatternBase {
    template<typename ...Args>
    Measure_Pattern(Args&& ...args) : QuantumPatternBase("Measure", 1, 2, std::forward<Args>(args)...) {}
    LogicalResult replace(hugr_mlir::ExtensionOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

struct QAlloc_Pattern : QuantumPatternBase {
    template<typename ...Args>
    QAlloc_Pattern(Args&& ...args) : QuantumPatternBase("QAlloc", 0, 1, std::forward<Args>(args)...) {}
    LogicalResult replace(hugr_mlir::ExtensionOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

struct QFree_Pattern : QuantumPatternBase {
    template<typename ...Args>
    QFree_Pattern(Args&& ...args) : QuantumPatternBase("QFree", 1, 0, std::forward<Args>(args)...) {}
    LogicalResult replace(hugr_mlir::ExtensionOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

}

SmallVector<QIR_FuncData> QIR_FuncDataFactory::all() const {
    SmallVector<QIR_FuncData> r;
    for(auto i = 0u; i < static_cast<unsigned>(QIR_FuncIndex::_END); ++i) {
        r.push_back(get(QIR_FuncIndex(i)));
    }
    return r;
}

QIR_FuncData QIR_FuncDataFactory::get(QIR_FuncIndex i) const {
    OpBuilder rw(context);
    switch(i) {
       case QIR_FuncIndex::H: return QIR_FuncData{"__quantum__qis__h__body", rw.getFunctionType(getPointerType(), {})};
       case QIR_FuncIndex::Measure: return QIR_FuncData{"__quantum__qis__m__body", rw.getFunctionType(getPointerType(), getPointerType())};
       case QIR_FuncIndex::QAlloc: return QIR_FuncData{"__quantum__rt__qubit__allocate", rw.getFunctionType({}, getPointerType())};
       case QIR_FuncIndex::QFree: return QIR_FuncData{"__quantum__rt__qubit__release", rw.getFunctionType(getPointerType(), {})};
       case QIR_FuncIndex::QIR_Result0: return QIR_FuncData{"__quantum__rt__result_get_zero", rw.getFunctionType({}, getPointerType())};
       case QIR_FuncIndex::QIR_Result1: return QIR_FuncData{"__quantum__rt__result_get_one", rw.getFunctionType({}, getPointerType())};
       case QIR_FuncIndex::QIR_ResultEq: return QIR_FuncData{"__quantum__rt__result_equal", rw.getFunctionType({getPointerType(), getPointerType()}, rw.getIndexType())};
       default: return {};
    }
}

LogicalResult H_Pattern::replace(hugr_mlir::ExtensionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rw) const {
    auto arg1 = adaptor.getArgs()[0];
    auto fd = funcdata.get(QIR_FuncIndex::H);
    rw.replaceAllUsesWith(op.getResult(0), arg1);
    rw.replaceOpWithNewOp<func::CallOp>(op, fd.symbol, fd.type.getResults(), arg1);
    return success();
}

static Value itobool(RewriterBase& rw, Location loc, Value i) {
  auto empty_tuple_t = rw.getType<TupleType>();
  auto sum_t = rw.getType<hugr_mlir::SumType>(SmallVector<Type>{empty_tuple_t, empty_tuple_t});
  auto empty_tuple_a = rw.getAttr<hugr_mlir::TupleAttr>(ArrayRef<TypedAttr>{});
  auto true_a = hugr_mlir::SumAttr::get(sum_t, 1, empty_tuple_a);
  auto false_a = hugr_mlir::SumAttr::get(sum_t, 0, empty_tuple_a);

  auto true_v = rw.createOrFold<hugr_mlir::ConstantOp>(loc, true_a);
  auto false_v = rw.createOrFold<hugr_mlir::ConstantOp>(loc, false_a);
  return rw.createOrFold<arith::SelectOp>(loc, i, true_v, false_v);
}

LogicalResult Measure_Pattern::replace(hugr_mlir::ExtensionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rw) const {
    auto arg1 = adaptor.getArgs()[0];
    auto measure_fd = funcdata.get(QIR_FuncIndex::Measure);
    auto r0_fd = funcdata.get(QIR_FuncIndex::QIR_Result0);
    auto r1_fd = funcdata.get(QIR_FuncIndex::QIR_Result1);
    auto req_fd = funcdata.get(QIR_FuncIndex::QIR_ResultEq);
    auto r = rw.create<func::CallOp>(op.getLoc(), measure_fd.symbol, measure_fd.type.getResults(), adaptor.getArgs());
    auto r1 = rw.create<func::CallOp>(op.getLoc(), r1_fd.symbol, r1_fd.type.getResults(), ValueRange{});
    auto r_eq_1_op = rw.create<func::CallOp>(op.getLoc(), req_fd.symbol, req_fd.type.getResults(), ValueRange{r.getResult(0), r1.getResult(0)});
    auto index_0 = rw.createOrFold<index::ConstantOp>(op.getLoc(), 0);
    assert(r_eq_1_op.getResult(0).getType() == index_0.getType() && "must");
    auto bool_r = itobool(rw, op.getLoc(), rw.createOrFold<index::CmpOp>(op.getLoc(), index::IndexCmpPredicate::NE, r_eq_1_op.getResult(0), index_0));
    rw.replaceOp(op, SmallVector<Value>{arg1, bool_r});
    return success();
}

LogicalResult QAlloc_Pattern::replace(hugr_mlir::ExtensionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rw) const {
    auto fd = funcdata.get(QIR_FuncIndex::QAlloc);
    rw.replaceOpWithNewOp<func::CallOp>(op, fd.symbol, fd.type.getResults());
    return success();
}

LogicalResult QFree_Pattern::replace(hugr_mlir::ExtensionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rw) const {
    auto fd = funcdata.get(QIR_FuncIndex::QFree);
    rw.replaceOpWithNewOp<func::CallOp>(op, fd.symbol, fd.type.getResults(), adaptor.getArgs());
    return success();
}

LogicalResult ConvertHugrExtQuantumPass::initialize(MLIRContext * context) {
    type_converter = std::make_shared<QuantumTypeConverter>();

    RewritePatternSet ps(context);
    ps.add<H_Pattern,Measure_Pattern,QAlloc_Pattern,QFree_Pattern>(*type_converter, context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(ps, *type_converter);
    populateBranchOpInterfaceTypeConversionPattern(ps, *type_converter);
    populateReturnOpTypeConversionPattern(ps, *type_converter);
    populateCallOpTypeConversionPattern(ps, *type_converter);

    patterns = FrozenRewritePatternSet(std::move(ps), disabledPatterns, enabledPatterns);
    return success();
}

void ConvertHugrExtQuantumPass::runOnOperation() {
    auto* context = &getContext();
    auto module = getOperation();
    IRRewriter rw(context);
    rw.setInsertionPointToEnd(module.getBody());
    for(auto d: QIR_FuncDataFactory(context).all()) {
        if(module.lookupSymbol(d.symbol)) { continue; }
        auto func = rw.create<func::FuncOp>(rw.getUnknownLoc(), d.symbol, d.type);
        rw.updateRootInPlace(func, [&] { func.setSymVisibility("private"); });
    }

    ConversionTarget target(*context);
    target.addLegalDialect<>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([this](func::FuncOp op) {
        return type_converter->isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<hugr_mlir::ExtensionOp>([](hugr_mlir::ExtensionOp op) {
        return !op.getExtension().getName().starts_with("quantum.tket2");
    });
    target.addDynamicallyLegalDialect<hugr_mlir::HugrDialect,func::FuncDialect,index::IndexDialect,arith::ArithDialect>([this](Operation* op) -> bool {
        if(auto bo = llvm::dyn_cast<BranchOpInterface>(op)) {
            return isLegalForBranchOpInterfaceTypeConversionPattern(op, *type_converter);
        } else if (auto r = llvm::dyn_cast<func::ReturnOp>(op)) {
            return isLegalForReturnOpTypeConversionPattern(op, *type_converter);
        }
        return true;
    });
    target.addDynamicallyLegalOp<func::CallOp,func::CallIndirectOp>([this](Operation* op) -> bool {
        auto call = llvm::cast<CallOpInterface>(op);
        return type_converter->isSignatureLegal(FunctionType::get(op->getContext(), call.getArgOperands().getTypes(), call->getResultTypes()));
    });


    if(failed(applyPartialConversion(module, target, patterns))) {
        emitError(module.getLoc()) << "ConvertHugrExtQuantumPass: Failed to applyPartialConversion";
        return signalPassFailure();
    };





  if(hugrVerify) {
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<hugr_mlir::ExtensionOp>([](hugr_mlir::ExtensionOp op) {
      return !op.getExtension().getName().starts_with("quantum.tket2");
    });
    SmallVector<Operation*> bad_ops;
    getOperation()->walk([&](Operation* op) {
      if(target.isIllegal(op)) { bad_ops.push_back(op); }
    });
    if(!bad_ops.empty()) {
      auto ifd = emitError(getOperation()->getLoc(), "Failed to eliminate quantum ext_ops:");
      for(auto o: bad_ops) {
        ifd.attachNote(o->getLoc());
      }
      return signalPassFailure();
    }
  }
}
