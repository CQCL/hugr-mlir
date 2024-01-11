#include "hugr-mlir/Conversion/ConvertHugrFuncPass.h"

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

namespace hugr_mlir {
#define GEN_PASS_DEF_PRECONVERTHUGRFUNCPASS
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

static bool isTopLevelFunc(hugr_mlir::FuncOp op) {
  return llvm::isa_and_present<hugr_mlir::ModuleOp>(op->getParentOp());
}

namespace {

using namespace mlir;

struct PreConvertHugrFuncPass
    : hugr_mlir::impl::PreConvertHugrFuncPassBase<PreConvertHugrFuncPass> {
  using PreConvertHugrFuncPassBase::PreConvertHugrFuncPassBase;
  using FuncClosureMap_t =
      llvm::DenseMap<StringAttr, hugr_mlir::AllocFunctionOp>;
  // LogicalResult initialize(MLIRContext*) override;
  void runOnOperation() override;
};

struct ConvertHugrFuncPass
    : hugr_mlir::impl::ConvertHugrFuncPassBase<ConvertHugrFuncPass> {
  using ConvertHugrFuncPassBase::ConvertHugrFuncPassBase;
  void runOnOperation() override;
};

struct ClosureiseCallOp : OpConversionPattern<hugr_mlir::CallOp> {
  template <typename... Args>
  ClosureiseCallOp(
      PreConvertHugrFuncPass::FuncClosureMap_t& fcm, Args&&... args)
      : OpConversionPattern(std::forward<Args>(args)...),
        func_closure_map(fcm) {}
  LogicalResult matchAndRewrite(
      hugr_mlir::CallOp, OpAdaptor, ConversionPatternRewriter&) const override;

 private:
  PreConvertHugrFuncPass::FuncClosureMap_t& func_closure_map;
};

struct ClosureiseLoadConstantOp
    : OpConversionPattern<hugr_mlir::LoadConstantOp> {
  template <typename... Args>
  ClosureiseLoadConstantOp(
      PreConvertHugrFuncPass::FuncClosureMap_t& fcm, Args&&... args)
      : OpConversionPattern(std::forward<Args>(args)...),
        func_closure_map(fcm) {}
  LogicalResult matchAndRewrite(
      hugr_mlir::LoadConstantOp, OpAdaptor,
      ConversionPatternRewriter&) const override;

 private:
  PreConvertHugrFuncPass::FuncClosureMap_t& func_closure_map;
};

struct ClosureiseConstantOp : OpConversionPattern<hugr_mlir::ConstantOp> {
  template <typename... Args>
  ClosureiseConstantOp(
      PreConvertHugrFuncPass::FuncClosureMap_t& fcm, Args&&... args)
      : OpConversionPattern(std::forward<Args>(args)...),
        func_closure_map(fcm) {}
  LogicalResult matchAndRewrite(
      hugr_mlir::ConstantOp, OpAdaptor,
      ConversionPatternRewriter&) const override;

 private:
  PreConvertHugrFuncPass::FuncClosureMap_t& func_closure_map;
};

struct LowerSwitch : OpConversionPattern<hugr_mlir::SwitchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::SwitchOp, OpAdaptor,
      ConversionPatternRewriter&) const override;
};

struct HugrFuncToFunc : OpConversionPattern<hugr_mlir::FuncOp> {
  template <typename... Args>
  HugrFuncToFunc(PreConvertHugrFuncPass::FuncClosureMap_t& fcm, Args&&... args)
      : OpConversionPattern(std::forward<Args>(args)...),
        func_closure_map(fcm) {}
  LogicalResult matchAndRewrite(
      hugr_mlir::FuncOp, OpAdaptor, ConversionPatternRewriter&) const override;

 private:
  PreConvertHugrFuncPass::FuncClosureMap_t& func_closure_map;
};

struct HugrCallToCall : OpConversionPattern<hugr_mlir::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      hugr_mlir::CallOp, OpAdaptor, ConversionPatternRewriter&) const override;
};

struct HugrCallTopLevelToCall : OpConversionPattern<hugr_mlir::CallOp> {
  template <typename... Args>
  HugrCallTopLevelToCall(
      PreConvertHugrFuncPass::FuncClosureMap_t& fcm, Args&&... args)
      : OpConversionPattern(std::forward<Args>(args)...),
        func_closure_map(fcm) {}
  LogicalResult matchAndRewrite(
      hugr_mlir::CallOp, OpAdaptor, ConversionPatternRewriter&) const override;

 private:
  PreConvertHugrFuncPass::FuncClosureMap_t& func_closure_map;
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

LogicalResult ClosureiseConstantOp::matchAndRewrite(
    hugr_mlir::ConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rw) const {
  auto a = llvm::dyn_cast<hugr_mlir::StaticEdgeAttr>(op.getValue());
  if (!a || !llvm::isa<hugr_mlir::FunctionType>(a.getType())) {
    return failure();
  }
  auto v = func_closure_map.lookup(a.getRef().getLeafReference());
  if (!v) {
    return failure();
  }
  rw.setInsertionPoint(op);
  rw.replaceOp(op, rw.getRemappedValue(v));
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
  if (auto v = func_closure_map.lookup(a.getRef().getLeafReference())) {
    rw.replaceOp(op, rw.getRemappedValue(v));
  } else {
    auto memref_t = MemRefType::get({}, rw.getType<hugr_mlir::ClosureType>());
    auto func = rw.createOrFold<func::ConstantOp>(op.getLoc(), getTypeConverter()->convertType(a.getType()), a.getRef().getLeafReference());
    auto closure = rw.createOrFold<ub::PoisonOp>(op.getLoc(), memref_t, rw.getAttr<ub::PoisonAttr>());
    rw.replaceOp(op, SmallVector{func,closure});
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

  auto callee = func_closure_map.lookup(attr.getRef().getLeafReference());
  if (!callee) {
    return failure();
  }

  rw.replaceOpWithNewOp<hugr_mlir::CallOp>(op, callee, op.getInputs());
  return mlir::success();
}

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

    if(auto alloc_func_op = func_closure_map.lookup(op.getSymNameAttr())) {
      SmallVector<Attribute> attrs;
      llvm::transform(op.getCaptures().getTypes(), std::back_inserter(attrs), [](auto x) { return TypeAttr::get(x); });
      rw.updateRootInPlace(alloc_func_op, [&] {
        alloc_func_op.setClosureTypesAttr(rw.getArrayAttr(attrs));
      });
      if (!op.getCaptures().empty()) {
        rw.setInsertionPoint(op);
        SmallVector<Value> unpacked;
        rw.createOrFold<hugr_mlir::UnpackTupleOp>(unpacked, loc, rw.getRemappedValue(alloc_func_op.getOutput()));
        assert(unpacked.size() == 2 && "must");
        rw.replaceOpWithNewOp<hugr_mlir::WriteClosureOp>(op, unpacked[1], adaptor.getCaptures());
      } else {
        rw.eraseOp(op);
      }
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
  auto loc = op.getLoc();
  rw.setInsertionPoint(op);
  SmallVector<Value> replacements;
  auto res_t = llvm::cast<TupleType>(
      getTypeConverter()->convertType(op.getOutput().getType()));
  assert(res_t.size() == 2 && "conversion ensures it");
  auto func = rw.createOrFold<func::ConstantOp>(
      op.getLoc(), res_t.getType(0), op.getFunc().getRef().getLeafReference());
  auto memref_t = MemRefType::get({}, rw.getType<hugr_mlir::ClosureType>());
  auto closure = rw.createOrFold<memref::AllocOp>(op.getLoc(), memref_t);
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
  if (!a) {
    return failure();
  }
  if (func_closure_map.lookup(a.getRef().getLeafReference())) {
    return rw.notifyMatchFailure(op, "func_closure_map does hold this callee");
  }

  SmallVector<Type> return_types;
  if (failed(getTypeConverter()->convertTypes(
          llvm::cast<hugr_mlir::FunctionType>(a.getType()).getResultTypes(),
          return_types))) {
    return failure();
  }
  rw.setInsertionPoint(op);
  auto memref_t = MemRefType::get({}, rw.getType<hugr_mlir::ClosureType>());
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

void PreConvertHugrFuncPass::runOnOperation() {
  auto context = &getContext();
  // collect funcs
  llvm::SmallVector<hugr_mlir::FuncOp> hugr_funcs;
  getOperation()->walk<WalkOrder::PostOrder>(
      [&hugr_funcs](hugr_mlir::FuncOp op) { hugr_funcs.push_back(op); });

  // alloc_closure at the region entry block for each func, map sym -> value
  FuncClosureMap_t func_closures;
  IRRewriter rw(context);
  for (auto f : hugr_funcs) {
    if (isTopLevelFunc(f)) {
      continue;
    }
    rw.setInsertionPointToStart(&f->getParentRegion()->front());
    auto o = rw.create<hugr_mlir::AllocFunctionOp>(
        f.getLoc(), f.getStaticEdgeAttr());
    func_closures.insert({f.getSymNameAttr(), o});
  }

  // legalize calls to use a closure
  ConversionTarget target(*context);
  target.addLegalDialect<
      cf::ControlFlowDialect, hugr_mlir::HugrDialect, index::IndexDialect,
      func::FuncDialect>();
  target.addIllegalOp<hugr_mlir::SwitchOp>();
  target.addDynamicallyLegalOp<hugr_mlir::CallOp>([&](hugr_mlir::CallOp op) {
    return !!op.getCalleeValue() ||
           !func_closures.contains(
               op.getCalleeAttrAttr().getRef().getLeafReference());
  });
  target.addDynamicallyLegalOp<hugr_mlir::LoadConstantOp>(
      [&](hugr_mlir::LoadConstantOp op) {
        return !llvm::isa<hugr_mlir::FunctionType>(op.getConstRef().getType());
      });
  target.addDynamicallyLegalOp<hugr_mlir::ConstantOp>(
      [&](hugr_mlir::ConstantOp op) {
        return !llvm::isa<hugr_mlir::FunctionType>(op.getValue().getType()) ||
               !func_closures.contains(
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
      emitError(getOperation()->getLoc()) << "Failed to closureize call ops";
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

void ConvertHugrFuncPass::runOnOperation() {
  auto* context = &getContext();

  PreConvertHugrFuncPass::FuncClosureMap_t fcm;
  getOperation()->walk([&fcm](hugr_mlir::AllocFunctionOp
                                                        op) {
    fcm.insert({op.getFuncAttr().getRef().getLeafReference(), op});
    // todo fail if insert fails
  });

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
