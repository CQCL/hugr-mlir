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
struct QuantumTypeConverter;
using namespace mlir;

struct QIR_FuncData {
    // QIR_FuncData(QIR_FuncData const&) = default;
    // QIR_FuncData& operator=(QIR_FuncData const&) = default;
    StringRef symbol{};
    FunctionType type{nullptr};
};

enum class QIR_FuncIndex : unsigned {
  H, X, Z,
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
    // template<QIR_FuncIndex>
    // LogicalResult build(QuantumTypeConverter const&, ConversionPatternRewriter& rw, Location loc, hugr_mlir::ExtensionOp, hugr_mlir::ExtensionOp::Adaptor) const;
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

    static hugr_mlir::OpaqueType getQubitType(OpBuilder& rw) {
        return rw.getType<hugr_mlir::OpaqueType>("qubit", rw.getAttr<hugr_mlir::ExtensionAttr>("prelude"), ArrayRef<Attribute>{}, hugr_mlir::TypeConstraint::Linear);

    }
    static bool is_qubit(Type t)  {
        OpBuilder rw(t.getContext());
        return t == getQubitType(rw);
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

struct X_Pattern : QuantumPatternBase {
    template<typename ...Args>
    X_Pattern(Args&& ...args) : QuantumPatternBase("Measure", 1, 2, std::forward<Args>(args)...) {}
    LogicalResult replace(hugr_mlir::ExtensionOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

struct Z_Pattern : QuantumPatternBase {
    template<typename ...Args>
    Z_Pattern(Args&& ...args) : QuantumPatternBase("Measure", 1, 2, std::forward<Args>(args)...) {}
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

// template<QIR_FuncIndex func_index>
// LogicalResult QIR_FuncDataFactory::build(QuantumTypeConverter const& tc, ConversionPatternRewriter& rw, Location loc, hugr_mlir::ExtensionOp op, hugr_mlir::ExtensionOp::Adaptor adaptor) const {
//     auto func_data = get(func_index);
//     if(adaptor.getArgs().getTypes() != func_data.type.getInputs()) {
//         return rw.notifyMatchFailure(op, [&](Diagnostic& d) {
//             d << "Type mismatch in args of quantum extension op. Expected(" << func_data.type.getInputs() << "), Found (" << adaptor.getArgs().getTypes();
//         });
//     }
//     SmallVector<Type> converted_res_tys;
//     if(failed(tc.convertTypes(op.getResultTypes(), converted_res_tys))) {
//         return rw.notifyMatchFailure(op, [&](Diagnostic& d) {
//             d << "Failed to convert result types:" << op.getResultTypes();
//         });
//     }
//     if(converted_res_tys != func_data.type.getResults()) {
//         return rw.notifyMatchFailure(op, [&](Diagnostic& d) {
//             d << "Type mismatch in results of quantum extension op. Expected(" << func_data.type.getResults() << "), Found (" << converted_res_tys;
//         });
//     }
//     SmallVector<Value> qubit_args;
//     SmallVector<Value> qubit_results;
//     llvm::copy_if(op.getArgs(), std::back_inserter(qubit_args), [](auto x) { return QuantumTypeConverter::is_qubit(x.getType()); });
//     llvm::copy_if(op.getResults(), std::back_inserter(qubit_results), [](auto x) { return QuantumTypeConverter::is_qubit(x.getType()); });
//     if(qubit_args.size() != qubit_results.size()) {
//         return rw.notifyMatchFailure(op, [&](Diagnostic& d) {
//             d << "Mismatch in qubit args and results: " << qubit_args.size() << " != " << qubit_results.size();
//         });
//     }
//     SmallVector<Value> ptr_args;
//     llvm::transform(qubit_args, std::back_inserter(ptr_args), [&](auto x) {return rw.getRemappedValue(x);});
//     rw.replaceAllUsesWith(qubit_results, ptr_args);





//     return success();
// }

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
       case QIR_FuncIndex::X: return QIR_FuncData{"__quantum__qis__x__body", rw.getFunctionType(getPointerType(), {})};
       case QIR_FuncIndex::Z: return QIR_FuncData{"__quantum__qis__z__body", rw.getFunctionType(getPointerType(), {})};
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

LogicalResult Z_Pattern::replace(hugr_mlir::ExtensionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rw) const {
    auto arg1 = adaptor.getArgs()[0];
    auto fd = funcdata.get(QIR_FuncIndex::Z);
    rw.replaceAllUsesWith(op.getResult(0), arg1);
    rw.replaceOpWithNewOp<func::CallOp>(op, fd.symbol, fd.type.getResults(), arg1);
    return success();
}

LogicalResult X_Pattern::replace(hugr_mlir::ExtensionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rw) const {
    auto arg1 = adaptor.getArgs()[0];
    auto fd = funcdata.get(QIR_FuncIndex::X);
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
