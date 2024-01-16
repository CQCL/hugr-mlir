#include "hugr-mlir/Conversion/ConvertHugrExtArithPass.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hugr_mlir {
#define GEN_PASS_DEF_CONVERTHUGREXTARITHPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}


namespace {
using namespace mlir;

static DenseSet<StringRef> getIntOpsUnimplementedOps() {
  return {"inarrow_u", "inarrow_s", "itobool", "ifrombool",
  "idivmod_checked_u",  "idivmod_checked_s", "idivmod_u",  "idivmod_s",
  "idiv_checked_u",  "idiv_checked_s",
  "imod_checked_u",  "imod_checked_s",
  "inot", "irotl", "irotr",
  };
};

static DenseSet<StringRef> getFloatOpsUnimplementedOps() {
  return {"fmax", "fmin",};
};

static DenseSet<StringRef> getConversionOpsUnimplementedOps() {
  return {"trunc_u", "trunc_s"};
};

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

static Value booltoi(RewriterBase& rw, Location loc, Value i) {
  return rw.createOrFold<index::CastUOp>(loc, rw.getI1Type(), rw.createOrFold<hugr_mlir::ReadTagOp>(loc, i));
}

struct ConvertHugrExtArithPass : hugr_mlir::impl::ConvertHugrExtArithPassBase<ConvertHugrExtArithPass> {
  using ConvertHugrExtArithPassBase::ConvertHugrExtArithPassBase;
  void runOnOperation() override;
  LogicalResult initialize(MLIRContext* context) override;
private:
  FrozenRewritePatternSet patterns;
};


template<char* name>
void replace_extension(hugr_mlir::ExtensionOp op, PatternRewriter& rw);

struct HugrExtensionOpPattern : OpRewritePattern<hugr_mlir::ExtensionOp> {
  template<typename ...Args>
  HugrExtensionOpPattern(StringRef extname_, StringRef opname_, int num_args_, int num_results_, Args&& ...args) :
    opname(opname_), extname(extname_), num_args(num_args_), num_results(num_results_),
    OpRewritePattern(std::forward<Args>(args)...) {
  }

  LogicalResult matchAndRewrite(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const override {
    if(!op.getExtension().getName().equals(extname) || !op.getHugrOpname().equals(opname) || op.getNumOperands() != num_args || op.getNumResults() != num_results) {
      return rw.notifyMatchFailure(op, [&](auto& d) {
        d << "Expected (" << opname << "," << extname << "," << num_args << "," << num_results << ")\n";
        d << "Found (" << op.getHugrOpname() << "," << op.getExtension().getName() << "," << op.getNumOperands() << "," << op.getNumResults() << ")\n";
      });
    }
    replace(op, rw);
    return success();
  }
  virtual void replace(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const = 0;
protected:
  StringRef const opname;
  StringRef const extname;
  int const num_args;
  int const num_results;
};

struct HugrExtensionOpPattern_float : HugrExtensionOpPattern {
  template<typename ...Args>
  HugrExtensionOpPattern_float(Args&& ...args) : HugrExtensionOpPattern("arithmetic.float", std::forward<Args>(args)...) {
    if(getFloatOpsUnimplementedOps().contains(opname)) {
      emitWarning(UnknownLoc::get(getContext())) << "HugrExtensionOpPattern_float:opname is recorded as unimplemented but this is the implementation:" << opname;
    }
  }
};

struct HugrExtensionOpPattern_int : HugrExtensionOpPattern {
  template<typename ...Args>
  HugrExtensionOpPattern_int(Args&& ...args) : HugrExtensionOpPattern("arithmetic.int", std::forward<Args>(args)...) {
    if(getIntOpsUnimplementedOps().contains(opname)) {
      emitWarning(UnknownLoc::get(getContext())) << "HugrExtensionOpPattern_int:opname is recorded as unimplemented but this is the implementation:" << opname;
    }
  }
};

struct HugrExtensionOpPattern_conversion : HugrExtensionOpPattern {
  template<typename ...Args>
  HugrExtensionOpPattern_conversion(Args&& ...args) : HugrExtensionOpPattern("arithmetic.conversions", std::forward<Args>(args)...) {
    if(getConversionOpsUnimplementedOps().contains(opname)) {
      emitWarning(UnknownLoc::get(getContext())) << "HugrExtensionOpPattern_conversion:opname is recorded as unimplemented but this is the implementation:" << opname;
    }
  }
};

template<typename ReplacementOpT>
struct SimpleReplace_float_1a_1r : HugrExtensionOpPattern_float {
  template<typename ...Args>
  SimpleReplace_float_1a_1r(StringRef name, Args&& ...args) : HugrExtensionOpPattern_float(name, 1, 1, std::forward<Args>(args)...) {}
  void replace(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const override {
    rw.replaceOpWithNewOp<ReplacementOpT>(op, op.getResults()[0].getType(), op.getArgs()[0]);
  }
};

template<typename ReplacementOpT>
struct SimpleReplace_int_1a_1r : HugrExtensionOpPattern_int {
  template<typename ...Args>
  SimpleReplace_int_1a_1r(StringRef name, Args&& ...args) : HugrExtensionOpPattern_int(name, 1, 1, std::forward<Args>(args)...) {}
  void replace(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const override {
    rw.replaceOpWithNewOp<ReplacementOpT>(op, op.getResults()[0].getType(), op.getArgs()[0]);
  }
};

template<typename ReplacementOpT>
struct SimpleReplace_conversion_1a_1r : HugrExtensionOpPattern_conversion {
  template<typename ...Args>
  SimpleReplace_conversion_1a_1r(StringRef name, Args&& ...args) : HugrExtensionOpPattern_conversion(name, 1, 1, std::forward<Args>(args)...) {}
  void replace(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const override {
    rw.replaceOpWithNewOp<ReplacementOpT>(op, op.getResults()[0].getType(), op.getArgs()[0]);
  }
};

template<typename ReplacementOpT>
struct SimpleReplace_int_2a_1r : HugrExtensionOpPattern_int {
  template<typename ...Args>
  SimpleReplace_int_2a_1r(StringRef name, Args&& ...args) : HugrExtensionOpPattern_int(name, 2, 1, std::forward<Args>(args)...) {}
  void replace(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const override {
    rw.replaceOpWithNewOp<ReplacementOpT>(op, op.getResults()[0].getType(), op.getArgs()[0], op.getArgs()[0]);
  }
};

template<typename ReplacementOpT>
struct SimpleReplace_float_2a_1r : HugrExtensionOpPattern_float {
  template<typename ...Args>
  SimpleReplace_float_2a_1r(StringRef name, Args&& ...args) : HugrExtensionOpPattern_float(name, 2, 1, std::forward<Args>(args)...) {}
  void replace(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const override {
    rw.replaceOpWithNewOp<ReplacementOpT>(op, op.getResults()[0].getType(), op.getArgs()[0], op.getArgs()[0]);
  }
};

struct Replace_ineg : HugrExtensionOpPattern_int {
  template<typename ...Args>
  Replace_ineg(Args&& ...args) : HugrExtensionOpPattern_int("ineg", 1, 1, std::forward<Args>(args)...) {}
  void replace(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const override {
    auto t = op.getResult(0).getType();
    auto zero = rw.createOrFold<arith::ConstantOp>(op.getLoc(), t, rw.getZeroAttr(t));
    rw.replaceOpWithNewOp<arith::SubIOp>(op, zero, op.getOperand(0));
  }
};

struct Replace_icmp : HugrExtensionOpPattern_int {
  template<typename ...Args>
  Replace_icmp(StringRef name, arith::CmpIPredicate pred_, Args&& ...args) : HugrExtensionOpPattern_int(name, 2, 1, std::forward<Args>(args)...), pred(pred_) {}
  void replace(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const override {
    auto v = rw.createOrFold<arith::CmpIOp>(op.getLoc(), pred, op.getOperand(0), op.getOperand(1));
    v = itobool(rw, op.getLoc(), v);
    rw.replaceOp(op, v);
  }
protected:
  arith::CmpIPredicate const pred;
};

struct Replace_fcmp : HugrExtensionOpPattern_float {
  template<typename ...Args>
  Replace_fcmp(StringRef name, arith::CmpFPredicate pred_, Args&& ...args) : HugrExtensionOpPattern_float(name, 2, 1, std::forward<Args>(args)...), pred(pred_) {}
  void replace(hugr_mlir::ExtensionOp op, PatternRewriter& rw) const override {
    // TODO to match webassembly we should add fast math flag "nsz" (see llvm langref)
    auto v = rw.createOrFold<arith::CmpFOp>(op.getLoc(), pred, op.getOperand(0), op.getOperand(1));
    v = itobool(rw, op.getLoc(), v);
    rw.replaceOp(op, v);
  }
protected:
  arith::CmpFPredicate const pred;
};

}



LogicalResult ConvertHugrExtArithPass::initialize(MLIRContext* context) {
  RewritePatternSet ps(context);
  ps.add<SimpleReplace_int_1a_1r<arith::ExtUIOp>>("iwiden_u", context);
  ps.add<SimpleReplace_int_1a_1r<arith::ExtSIOp>>("iwiden_s", context);
  // TODO inarrow_{u,s} itobool ifrombool, ieq,ine,ilt_u,ilt_s,igt_u,igt_s,ile_u,ile_s,ige_u,ige_s

  ps.add<SimpleReplace_int_2a_1r<arith::MaxUIOp>>("imax_u", context);
  ps.add<SimpleReplace_int_2a_1r<arith::MaxSIOp>>("imax_s", context);
  ps.add<SimpleReplace_int_2a_1r<arith::MinUIOp>>("imin_u", context);
  ps.add<SimpleReplace_int_2a_1r<arith::MinSIOp>>("imin_s", context);
  ps.add<SimpleReplace_int_2a_1r<arith::AddIOp>>("iadd", context);
  ps.add<SimpleReplace_int_2a_1r<arith::SubIOp>>("isub", context);
  // TODO ineg
  ps.add<SimpleReplace_int_2a_1r<arith::MulIOp>>("imul", context);
  // TODO idivmod_checked_{u,s} idivmod_{u,s} idiv_checked_{u,s} imod_checked_{u,s}
  ps.add<SimpleReplace_int_2a_1r<arith::DivSIOp>>("idiv_s", context);
  ps.add<SimpleReplace_int_2a_1r<arith::DivUIOp>>("idiv_u", context);
  ps.add<SimpleReplace_int_2a_1r<arith::RemSIOp>>("imod_s", context);
  ps.add<SimpleReplace_int_2a_1r<arith::RemUIOp>>("imod_u", context);
  ps.add<SimpleReplace_int_1a_1r<math::AbsIOp>>("iabs", context);
  ps.add<SimpleReplace_int_2a_1r<arith::AndIOp>>("iand", context);
  ps.add<SimpleReplace_int_2a_1r<arith::OrIOp>>("ior", context);
  ps.add<SimpleReplace_int_2a_1r<arith::XOrIOp>>("ixor", context);
  // TODO inot
  ps.add<SimpleReplace_int_2a_1r<arith::ShLIOp>>("ishl", context);
  ps.add<SimpleReplace_int_2a_1r<arith::ShRUIOp>>("ishr", context);
  // TODO irot{l,r}
  // feq,fne,flt,fgt,fle,fge, fmax, fmin
  ps.add<SimpleReplace_float_2a_1r<arith::AddFOp>>("fadd", context);
  ps.add<SimpleReplace_float_2a_1r<arith::SubFOp>>("fsub", context);
  ps.add<SimpleReplace_float_1a_1r<arith::NegFOp>>("fneg", context);
  ps.add<SimpleReplace_float_1a_1r<math::AbsFOp>>("fabs", context);
  ps.add<SimpleReplace_float_2a_1r<arith::MulFOp>>("fmul", context);
  ps.add<SimpleReplace_float_2a_1r<arith::DivFOp>>("fdiv", context);
  ps.add<SimpleReplace_float_1a_1r<math::FloorOp>>("ffloor", context);
  ps.add<SimpleReplace_float_1a_1r<math::CeilOp>>("fceil", context);
  // TODO trunc_u, trunc_s
  ps.add<SimpleReplace_conversion_1a_1r<arith::SIToFPOp>>("convert_s", context);
  ps.add<SimpleReplace_conversion_1a_1r<arith::UIToFPOp>>("convert_u", context);

  ps.add<Replace_ineg>(context);
  ps.add<Replace_icmp>("ieq", arith::CmpIPredicate::eq, context);
  ps.add<Replace_icmp>("ine", arith::CmpIPredicate::ne, context);
  ps.add<Replace_icmp>("ilt_u", arith::CmpIPredicate::ult, context);
  ps.add<Replace_icmp>("ilt_s", arith::CmpIPredicate::slt, context);
  ps.add<Replace_icmp>("igt_u", arith::CmpIPredicate::ugt, context);
  ps.add<Replace_icmp>("igt_s", arith::CmpIPredicate::sgt, context);
  ps.add<Replace_icmp>("ige_u", arith::CmpIPredicate::uge, context);
  ps.add<Replace_icmp>("ige_s", arith::CmpIPredicate::sge, context);
  ps.add<Replace_icmp>("ile_u", arith::CmpIPredicate::ule, context);
  ps.add<Replace_icmp>("ile_s", arith::CmpIPredicate::sle, context);

  ps.add<Replace_fcmp>("feq", arith::CmpFPredicate::OEQ, context);
  ps.add<Replace_fcmp>("fne", arith::CmpFPredicate::ONE, context);
  ps.add<Replace_fcmp>("flt", arith::CmpFPredicate::OLT, context);
  ps.add<Replace_fcmp>("fgt", arith::CmpFPredicate::OGT, context);
  ps.add<Replace_fcmp>("fle", arith::CmpFPredicate::OLE, context);
  ps.add<Replace_fcmp>("fge", arith::CmpFPredicate::OGE, context);

  patterns = FrozenRewritePatternSet(std::move(ps), disabledPatterns, enabledPatterns);
  return success();
}

void ConvertHugrExtArithPass::runOnOperation() {
  GreedyRewriteConfig cfg;
  if(failed(applyPatternsAndFoldGreedily(getOperation(), patterns, cfg))) {
    emitError(getOperation()->getLoc(), "ConvertHugrExtArithPass: Failed to applyPatternsAndFoldGreedily");
    return signalPassFailure();
  }

  if(hugrVerify) {
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<hugr_mlir::ExtensionOp>([](hugr_mlir::ExtensionOp op) {
      return !op.getExtension().getName().starts_with("arithmetic");
    });
    SmallVector<Operation*> bad_ops;
    getOperation()->walk([&](Operation* op) {
      if(target.isIllegal(op)) { bad_ops.push_back(op); }
    });
    if(!bad_ops.empty()) {
      auto ifd = emitError(getOperation()->getLoc(), "Failed to eliminate arithmetic ext_ops:");
      for(auto o: bad_ops) {
        ifd.attachNote(o->getLoc());
      }
      return signalPassFailure();
    }
  }
}
