#include "hugr-mlir/Conversion/ConvertHugrFuncPass.h"
#include "hugr-mlir/Conversion/Utils.h"

#include "hugr-mlir/Conversion/TypeConverter.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"
#include "hugr-mlir/Transforms/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace hugr_mlir {
#define GEN_PASS_DEF_CONVERTHUGRFUNCPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir

static hugr_mlir::ModuleOp getOwningModule(mlir::Operation* op) {
  for (;;) {
    if (!op) {
      return nullptr;
    }
    if (auto r = llvm::dyn_cast<hugr_mlir::ModuleOp>(op)) {
      return r;
    }
    op = op->getParentOp();
  }
}


/* namespace { */
/* struct TopLevelHugrFuncOp : public hugr_mlir::FuncOp { */
/*   TopLevelHugrFuncOp(std::nullptr_t np) : hugr_mlir::FuncOp(np) {} */
/*   TopLevelHugrFuncOp(mlir::Operation* op) : hugr_mlir::FuncOp( */
/*     classof(op) ? FuncOp(op) : hugr_mlir::FuncOp(nullptr)) {} */
/*   /\* static bool classof(hugr_mlir::FuncOp op) { *\/ */
/*   /\*   return !!op && llvm::isa_and_present<hugr_mlir::ModuleOp>(op->getParentOp()); *\/ */
/*   /\* } *\/ */
/*   static bool classof(mlir::Operation * op) { */
/*     return classof(llvm::dyn_cast_if_present<hugr_mlir::FuncOp*>(op)); */
/*   } */
/*   static bool classof(hugr_mlir::FuncOp const* op) { */
/*     if(!op) { return false; } */
/*     auto x = const_cast<hugr_mlir::FuncOp*>(op)->getOperation(); */
/*     return x && llvm::isa_and_present<hugr_mlir::ModuleOp>(x->getParentOp()); */
/*   } */
/* }; */
/* } */

/* namespace llvm { */
/* template<typename To> */
/* struct CastInfo<To, TopLevelHugrFuncOp> : */
/*     CastInfo<To,  hugr_mlir::FuncOp> {}; */

/* template<> */
/* struct CastInfo<TopLevelHugrFuncOp, hugr_mlir::FuncOp> */
/*   : public NullableValueCastFailed<TopLevelHugrFuncOp> */
/*   , public DefaultDoCastIfPossible<TopLevelHugrFuncOp, hugr_mlir::FuncOp, CastInfo<TopLevelHugrFuncOp, hugr_mlir::FuncOp>> */
/*   , public CastIsPossible<TopLevelHugrFuncOp, hugr_mlir::FuncOp> { */
/*     static inline TopLevelHugrFuncOp doCast(hugr_mlir::FuncOp f) { return TopLevelHugrFuncOp(f); } */
/* }; */
/* template<typename From> */
/* struct CastInfo<TopLevelHugrFuncOp, From, CastIsPossible<hugr_mlir::FuncOp, From>> : NullableValueCastFailed<TopLevelHugrFuncOp> { */
/*   static inline bool isPossible(From op) { */
/*     return TopLevelHugrFuncOp::classof(llvm::dyn_cast_if_present<hugr_mlir::FuncOp>(op)); */
/*   } */
/*   static inline TopLevelHugrFuncOp doCast(From op) { return TopLevelHugrFuncOp(llvm::cast<hugr_mlir::FuncOp>(op)); } */

/* }; */

/* hugr_mlir::FuncOp, CastI> {} */
/*     CastInfo<hugr_mlir::FuncOp, mlir::Operation*>  { */
/*     static inline TopLevelHugrFuncOp castFailed() { return TopLevelHugrFuncOp(nullptr); } */
/* }; */
/*   using OpCastInfo = CastInfo<mlir::Operation*,From>; */
/*   static inline bool isPossible(From op) { */
/*     return TopLevelHugrFuncOp::classof(llvm::cast<mlir::Operation*>(op)); */
/*   } */
/*   static inline TopLevelHugrFuncOp doCast(From op) { return TopLevelHugrFuncOp(OpCastInfo::doCast(op)); } */
/*   static inline TopLevelHugrFuncOp castFailed() { return TopLevelHugrFuncOp(nullptr); } */
/*   /\* static TopLevelHugrFuncOp doCastIfPossible(From op) { return isPossible(op) ? TopLevelHugrFuncOp(OpCastInfo::doCast(op)) : TopLevelHugrFuncOp(nullptr); } *\/ */

/* }; */

/* } */

namespace {

using namespace mlir;


struct ConvertHugrFuncPass
    : hugr_mlir::impl::ConvertHugrFuncPassBase<ConvertHugrFuncPass> {
  using ConvertHugrFuncPassBase::ConvertHugrFuncPassBase;
  void runOnOperation() override;
};







struct HugrFuncToFunc : hugr_mlir::FuncClosureMapOpConversionPatternBase<hugr_mlir::FuncOp> {
  using FuncClosureMapOpConversionPatternBase::FuncClosureMapOpConversionPatternBase;
  LogicalResult matchAndRewrite(hugr_mlir::FuncOp, OpAdaptor, ConversionPatternRewriter&) const override;
};

struct HugrCallToCall : OpConversionPattern<hugr_mlir::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(hugr_mlir::CallOp, OpAdaptor, ConversionPatternRewriter&) const override;
};

struct HugrCallTopLevelToCall : hugr_mlir::FuncClosureMapOpConversionPatternBase<hugr_mlir::CallOp> {
  using FuncClosureMapOpConversionPatternBase::FuncClosureMapOpConversionPatternBase;
  LogicalResult matchAndRewrite(hugr_mlir::CallOp, OpAdaptor, ConversionPatternRewriter&) const override;
};

struct OutputToReturn : OpConversionPattern<hugr_mlir::OutputOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::OutputOp, OpAdaptor,
      ConversionPatternRewriter&) const override;
};

struct LowerAllocFunction : OpConversionPattern<hugr_mlir::AllocFunctionOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::AllocFunctionOp, OpAdaptor,
      ConversionPatternRewriter&) const override;
};

struct ConstToTopLevel : OpConversionPattern<hugr_mlir::ConstOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::ConstOp, OpAdaptor, ConversionPatternRewriter&) const override;
};

struct CallIndirectOpSignatureConversion
    : public OpConversionPattern<func::CallIndirectOp> {
  using OpConversionPattern::OpConversionPattern;

  /// Hook for derived classes to implement combined matching and rewriting.
  LogicalResult matchAndRewrite(
      func::CallIndirectOp callOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Convert the original function results.
    SmallVector<Type, 1> convertedResults;
    if (failed(typeConverter->convertTypes(
            callOp.getResultTypes(), convertedResults)))
      return failure();

    // If this isn't a one-to-one type mapping, we don't know how to aggregate
    // the results.
    if (callOp->getNumResults() != convertedResults.size()) return failure();

    // Substitute with the new result types from the corresponding FuncType
    // conversion.
    rewriter.replaceOpWithNewOp<func::CallIndirectOp>(
        callOp, adaptor.getCallee(), adaptor.getOperands());
    return success();
  }
};

}  // namespace


LogicalResult HugrFuncToFunc::matchAndRewrite(
    hugr_mlir::FuncOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rw) const {
  assert(getTypeConverter() && "must have type converter");
  if (op.isDeclaration() && op.getCaptures().size() > 0) {
    return rw.notifyMatchFailure(op, "A declaration with captures");
  }
  if (hugr_mlir::opHasNonLocalEdges(op)) {
    return rw.notifyMatchFailure(op, "nonlocal edges");
  }
  auto module = getOwningModule(op);
  if(!module) { return failure(); }

  auto loc = op.getLoc();

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
    // TODO we are throwing away arg attrs for closure args. probs fine, but we
    // could put them in this dics
    SmallVector<Attribute> new_arg_attrs{rw.getDictionaryAttr({})};
    llvm::copy(
        arg_attrs.getValue().drop_front(op.getCaptures().size()),
        std::back_inserter(new_arg_attrs));
    list.append(
        func::FuncOp::getArgAttrsAttrName(
            OperationName("func.func", getContext())),
        rw.getArrayAttr(new_arg_attrs));
  }

  auto converted_tuple_type = llvm::cast<TupleType>(
      getTypeConverter()->convertType(op.getFunctionType()));
  assert(converted_tuple_type.size() == 2 && "by conversion");
  auto converted_func_type =
      llvm::cast<FunctionType>(converted_tuple_type.getType(0));

  rw.setInsertionPointToEnd(&module.getBody().front());
  auto func = rw.create<func::FuncOp>(
      loc, op.getSymName(), converted_func_type, list.getAttrs());

  if (!op.getBody().empty()) {
    auto old_entry = &op.getBody().front();

    SmallVector<Location> new_arg_locs{loc};
    for (auto ba :
         old_entry->getArguments().drop_front(op.getCaptures().size())) {
      new_arg_locs.push_back(ba.getLoc());
    }
    assert(
        new_arg_locs.size() == converted_func_type.getNumInputs() &&
        "wrong num of locs");

    auto entry_block = rw.createBlock(
        &func.getBody(), func.getBody().begin(),
        converted_func_type.getInputs(), new_arg_locs);
    rw.setInsertionPointToStart(entry_block);
    SmallVector<Value> replacements;
    rw.createOrFold<hugr_mlir::ReadClosureOp>(
        replacements, loc, adaptor.getCaptures().getTypes(),
        entry_block->getArgument(0));
    llvm::copy(
        entry_block->getArguments().drop_front(),
        std::back_inserter(replacements));
    rw.inlineRegionBefore(func.getBody(), op.getBody(), func.getBody().end());
    rw.mergeBlocks(old_entry, entry_block, replacements);

    if(auto alloc_func_op = lookupAllocFunctionOp(SymbolRefAttr::get(op.getSymNameAttr()))) {
      rw.setInsertionPoint(op);
      SmallVector<Value> unpacked;
      rw.createOrFold<hugr_mlir::UnpackTupleOp>(unpacked, loc, rw.getRemappedValue(alloc_func_op.getOutput()));
      assert(unpacked.size() == 2 && "must");
      rw.replaceOpWithNewOp<hugr_mlir::WriteClosureOp>(op, unpacked[1], adaptor.getCaptures());
    } else {
      rw.eraseOp(op);
    };
  } else {
    assert(op.getCaptures().size() == 0 && "or we would have failed on entry");
    rw.eraseOp(op);
  }
  return success();
}

LogicalResult LowerAllocFunction::matchAndRewrite(
    hugr_mlir::AllocFunctionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rw) const {
  assert(getTypeConverter() && "must have type converter");
  if(!op.getClosureTypes().has_value()) { return failure(); }
  auto loc = op.getLoc();
  rw.setInsertionPoint(op);
  SmallVector<Value> replacements;
  auto res_t = llvm::cast<TupleType>(
      getTypeConverter()->convertType(op.getOutput().getType()));
  assert(res_t.size() == 2 && "conversion ensures it");
  auto func = rw.createOrFold<func::ConstantOp>(
      op.getLoc(), res_t.getType(0), op.getFunc().getRef().getLeafReference());
  auto memref_t = rw.getType<hugr_mlir::ClosureType>().getMemRefType();
  auto closure = op.getClosureTypes()->empty()
    ? rw.createOrFold<ub::PoisonOp>(op.getLoc(), memref_t, rw.getAttr<ub::PoisonAttr>())
    : rw.createOrFold<memref::AllocOp>(op.getLoc(), memref_t);

  SmallVector args{func, closure};
  rw.replaceOp(op, rw.createOrFold<hugr_mlir::MakeTupleOp>(loc, res_t, args));
  return success();
}

LogicalResult HugrCallToCall::matchAndRewrite(
    hugr_mlir::CallOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rw) const {
  auto v = adaptor.getCalleeValue();
  if (!v) {
    return failure();
  }

  auto tt = llvm::dyn_cast<TupleType>(v.getType());
  assert(
      tt && tt.size() == 2 && llvm::isa<FunctionType>(tt.getType(0)) &&
      llvm::isa<MemRefType>(tt.getType(1)) && "conversion not working");

  rw.setInsertionPoint(op);
  auto unpack = rw.create<hugr_mlir::UnpackTupleOp>(op.getLoc(), v);
  SmallVector<Value> args{unpack.getOutputs()[1]};
  llvm::copy(adaptor.getInputs(), std::back_inserter(args));
  rw.replaceOpWithNewOp<func::CallIndirectOp>(op, unpack.getOutputs()[0], args);
  return success();
}

LogicalResult HugrCallTopLevelToCall::matchAndRewrite(
    hugr_mlir::CallOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rw) const {
  auto a = adaptor.getCalleeAttrAttr();
  if (!a) { return failure(); }
  auto tlf = lookupTopLevelFunc(a.getRef());
  if (!tlf) {
    return rw.notifyMatchFailure(op, "not a call to a top level func");
  }

  SmallVector<Type> return_types;
  if (failed(getTypeConverter()->convertTypes(
          llvm::cast<hugr_mlir::FunctionType>(a.getType()).getResultTypes(),
          return_types))) {
    return failure();
  }
  rw.setInsertionPoint(op);
  auto memref_t = rw.getType<hugr_mlir::ClosureType>().getMemRefType();
  auto closure = rw.createOrFold<ub::PoisonOp>(
      op.getLoc(), memref_t, rw.getAttr<ub::PoisonAttr>());
  SmallVector<Value> args{closure};
  llvm::copy(adaptor.getInputs(), std::back_inserter(args));

  rw.replaceOpWithNewOp<func::CallOp>(op, a.getRef(), return_types, args);
  return success();
}

LogicalResult ConstToTopLevel::matchAndRewrite(hugr_mlir::ConstOp op, OpAdaptor adaptor, ConversionPatternRewriter &rw) const {
  auto mod = getOwningModule(op);
  if(!mod || op->getParentOp() == mod) { return failure(); }
  rw.setInsertionPointToEnd(&mod.getBody().front());
  rw.clone(*op);
  rw.eraseOp(op);
  return success();
}

LogicalResult OutputToReturn::matchAndRewrite(
    hugr_mlir::OutputOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rw) const {
  auto parent = op->getParentOp();
  if (!llvm::isa_and_nonnull<func::FuncOp>(parent)) {
    return failure();
  }
  rw.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOutputs());

  return success();
}

void ConvertHugrFuncPass::runOnOperation() {
  auto* context = &getContext();

  IRRewriter rw(context);
  SymbolTableCollection stc;
  hugr_mlir::FuncClosureMap fcm;
  llvm::DenseMap<StringRef,hugr_mlir::FuncOp> funcs;
  llvm::SmallVector<hugr_mlir::AllocFunctionOp> allocs;
  getOperation()->walk([&](Operation* op) {
    llvm::TypeSwitch<Operation*, void>(op)
      .Case([&](hugr_mlir::FuncOp op) { funcs.insert({op.getSymName(), op});})
      .Case([&](hugr_mlir::AllocFunctionOp op) { allocs.push_back(op);});
  });
  for(auto alloc: allocs) {
    auto func_op = funcs.lookup(alloc.getFunc().getRef().getLeafReference());
    if(!func_op) {
      emitError(alloc->getLoc()) << "Unknown symbol: " << alloc.getFunc().getRef();
      return signalPassFailure();
    }
    if(mlir::failed(fcm.insert(func_op, alloc))) {
      auto ifd = emitError(alloc->getLoc()) << "Failed to ingress hugr.alloc_function op";
      ifd.attachNote(func_op.getLoc());
      return signalPassFailure();
    }
    rw.updateRootInPlace(alloc,[&] {
      SmallVector<Attribute> attrs;
      llvm::transform(func_op.getCaptures().getTypes(), std::back_inserter(attrs), [](auto x) { return TypeAttr::get(x); });
      alloc.setClosureTypesAttr(rw.getArrayAttr(attrs));
    });
  }

  auto type_converter = hugr_mlir::createSimpleTypeConverter();
  type_converter->addConversion(
      [tc = type_converter.get()](
          hugr_mlir::FunctionType t) -> std::optional<Type> {
        OpBuilder rw(t.getContext());
        auto memref_t =
            MemRefType::get({}, rw.getType<hugr_mlir::ClosureType>());
        SmallVector<Type> new_arg_tys{memref_t};
        llvm::copy(t.getArgumentTypes(), std::back_inserter(new_arg_tys));
        return rw.getType<TupleType>(SmallVector<Type>{
            rw.getFunctionType(new_arg_tys, t.getResultTypes()),
            new_arg_tys[0]});
      });
  type_converter->addConversion(
      [tc = type_converter.get()](FunctionType ft) -> std::optional<Type> {
        TypeConverter::SignatureConversion sc1(ft.getNumInputs()),
            sc2(ft.getNumResults());
        if (failed(tc->convertSignatureArgs(ft.getInputs(), sc1)) ||
            failed(tc->convertSignatureArgs(ft.getResults(), sc2))) {
          return nullptr;
        }
        return {FunctionType::get(
            ft.getContext(), sc1.getConvertedTypes(), sc2.getConvertedTypes())};
      });

  RewritePatternSet ps(context);
  ps.add<HugrFuncToFunc, HugrCallTopLevelToCall>(fcm, *type_converter, context);
  ps.add<HugrCallToCall, LowerAllocFunction, OutputToReturn,ConstToTopLevel>(
      *type_converter, context);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      ps, *type_converter);
  populateBranchOpInterfaceTypeConversionPattern(ps, *type_converter);
  populateReturnOpTypeConversionPattern(ps, *type_converter);
  populateCallOpTypeConversionPattern(ps, *type_converter);
  ps.add<CallIndirectOpSignatureConversion>(*type_converter, context);

  FrozenRewritePatternSet patterns(
      std::move(ps), disabledPatterns, enabledPatterns);

  ConversionTarget target(*context);
  target.addLegalDialect<
      hugr_mlir::HugrDialect, func::FuncDialect, cf::ControlFlowDialect,
      index::IndexDialect, memref::MemRefDialect, ub::UBDialect>();
  target.addIllegalOp<
      hugr_mlir::FuncOp, hugr_mlir::CallOp, hugr_mlir::AllocFunctionOp,
      hugr_mlir::SwitchOp>();
  target.addDynamicallyLegalOp<hugr_mlir::LoadConstantOp>(
      [](hugr_mlir::LoadConstantOp op) {
        return !llvm::isa<hugr_mlir::FunctionType>(op.getConstRef().getType());
      });
  target.addDynamicallyLegalOp<hugr_mlir::ConstOp>(
      [](hugr_mlir::ConstOp op) {
        return llvm::isa_and_present<hugr_mlir::ModuleOp>(op->getParentOp());
      });
  target.addDynamicallyLegalOp<hugr_mlir::ConstantOp>(
      [](hugr_mlir::ConstantOp op) {
        return !llvm::isa<hugr_mlir::FunctionType>(op.getValue().getType());
      });
  target.addDynamicallyLegalOp<hugr_mlir::OutputOp>([](hugr_mlir::OutputOp op) {
    auto p = op->getParentOp();
    return !p || !llvm::isa<func::FuncOp>(p);
  });
  target.addDynamicallyLegalOp<func::FuncOp>(
      [tc = type_converter.get()](func::FuncOp op) {
        return tc->isSignatureLegal(op.getFunctionType());
      });
  target.addDynamicallyLegalOp<cf::BranchOp, cf::SwitchOp, cf::CondBranchOp>(
      [tc = type_converter.get()](Operation* op) {
        return isLegalForBranchOpInterfaceTypeConversionPattern(op, *tc);
      });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [tc = type_converter.get()](func::ReturnOp op) {
        return isLegalForReturnOpTypeConversionPattern(op, *tc);
      });
  target.addDynamicallyLegalOp<func::CallOp>(
      [tc = type_converter.get()](func::CallOp op) {
        return tc->isSignatureLegal(op.getCalleeType());
      });
  target.addDynamicallyLegalOp<func::CallIndirectOp>(
      [tc = type_converter.get()](func::CallIndirectOp op) {
        return tc->isSignatureLegal(op.getCallee().getType());
      });

  if (failed(applyPartialConversion(getOperation(), target, patterns))) {
    emitError(getOperation()->getLoc())
        << "ConvertHugrFuncPass:Failed to applyPartialConversion";
    return signalPassFailure();
  }
  {
    IRRewriter rw(&getContext());
    SmallVector<hugr_mlir::ModuleOp> worklist{
        getOperation().getOps<hugr_mlir::ModuleOp>()};
    for (auto hm : worklist) {
      if (!hm.getBody().empty()) {
        rw.inlineBlockBefore(
            &hm.getBody().front(), getOperation().getBody(),
            getOperation().getBody()->end());
      }
      rw.eraseOp(hm);
    }
  }
}
