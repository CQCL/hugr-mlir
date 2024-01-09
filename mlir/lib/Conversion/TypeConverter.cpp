#include "hugr-mlir/Conversion/TypeConverter.h"

#include "hugr-mlir/IR/HugrOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
// #include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

namespace {
using namespace mlir;

struct SimpleHugrTypeConverter : TypeConverter {
  SimpleHugrTypeConverter();
  FailureOr<Value> materializeTargetSum(
      OpBuilder&, hugr_mlir::SumType, Value, Location) const;
  FailureOr<Value> materializeTargetTuple(
      OpBuilder&, TupleType, Value, Location) const;
  FailureOr<Value> materializeSourceSum(
      OpBuilder&, hugr_mlir::SumType, TypedValue<hugr_mlir::SumType>,
      Location) const;
  FailureOr<Value> materializeSourceTuple(
      OpBuilder&, TupleType, TypedValue<TupleType>, Location) const;
};

struct HugrTypeConverter : OneToNTypeConverter {
  HugrTypeConverter();
  FailureOr<OneToNTypeMapping> partitionSumType(hugr_mlir::SumType) const;
  FailureOr<OneToNTypeMapping> partitionTupleType(TupleType) const;
  FailureOr<SmallVector<Value>> materializeTargetSum(
      OpBuilder&, TypeRange, Value, Location) const;
  FailureOr<SmallVector<Value>> materializeTargetTuple(
      OpBuilder&, TypeRange, Value, Location) const;
  FailureOr<Value> materializeSourceSum(
      OpBuilder&, hugr_mlir::SumType, ValueRange, Location) const;
  FailureOr<Value> materializeSourceTuple(
      OpBuilder&, TupleType, ValueRange, Location) const;
};

}  // namespace

// void hugr_mlir::populateHugrToLLVMConversionPatterns(mlir::RewritePatternSet&
// ps, mlir::TypeConverter const& tc, int benefit) {

//     ps.add<ConvertHugrFuncToFuncConversionPattern>(tc, ps.getContext(),
//     benefit);
// }
mlir::FailureOr<mlir::OneToNTypeMapping> HugrTypeConverter::partitionSumType(
    hugr_mlir::SumType st) const {
  if (st.numAlts() == 0) {
    return failure();
  }
  OneToNTypeMapping r(st.getTypes());
  if (failed(convertSignatureArgs(st.getTypes(), r))) {
    return failure();
  }
  return success(r);
}

mlir::FailureOr<mlir::OneToNTypeMapping> HugrTypeConverter::partitionTupleType(
    TupleType tt) const {
  OneToNTypeMapping r(tt.getTypes());
  if (failed(convertSignatureArgs(tt.getTypes(), r))) {
    return failure();
  }
  return success(r);
}

mlir::FailureOr<mlir::SmallVector<mlir::Value>>
HugrTypeConverter::materializeTargetSum(
    OpBuilder& rw, TypeRange ts, Value src, Location loc) const {
  auto st = llvm::dyn_cast<hugr_mlir::SumType>(src.getType());
  if (!st) {
    return failure();
  }

  auto mb_partition = partitionSumType(st);
  if (failed(mb_partition)) {
    return failure();
  }

  SmallVector<Type> expected_tys{rw.getIndexType()};
  llvm::copy(
      mb_partition->getConvertedTypes(), std::back_inserter(expected_tys));
  if (expected_tys != ts) {
    return failure();
  }

  auto const tag = rw.createOrFold<hugr_mlir::ReadTagOp>(loc, src);
  SmallVector<SmallVector<Value>> undef_vs;
  for (auto i = 0; i < mb_partition->getOriginalTypes().size(); ++i) {
    auto& vec = undef_vs.emplace_back();
    llvm::transform(
        mb_partition->getConvertedTypes(i), std::back_inserter(vec),
        [&rw, loc](auto t) {
          return rw.createOrFold<ub::PoisonOp>(
              loc, t, rw.getAttr<ub::PoisonAttr>());
        });
  };

  SmallVector<int64_t> cases;
  for (auto i = 1; i < st.numAlts(); ++i) {
    cases.push_back(i);
  }

  auto switch_op =
      rw.create<scf::IndexSwitchOp>(loc, ts, tag, cases, cases.size());
  for (auto r : switch_op.getRegions()) {
    assert(r->empty() && "Region should not have been constructed with block");
    auto n = r->getRegionNumber();
    auto src_t = st.getAltType(n);

    rw.setInsertionPointToStart(rw.createBlock(r));
    SmallVector<Value> rs{tag};
    for (auto i = 0; i < mb_partition->getOriginalTypes().size(); ++i) {
      if (i != n) {
        llvm::copy(undef_vs[i], std::back_inserter(rs));
      } else {
        auto variant_v =
            rw.createOrFold<hugr_mlir::ReadVariantOp>(loc, src_t, src, n);
        auto mb_converted_variant_v = this->materializeTargetConversion(
            rw, loc, mb_partition->getConvertedTypes(i), variant_v);
        assert(mb_converted_variant_v && "already checked this will work");
        llvm::copy(*mb_converted_variant_v, std::back_inserter(rs));
      }
    }
    rw.create<scf::YieldOp>(loc, rs);
  }
  return success(switch_op.getResults());
}

mlir::FailureOr<mlir::SmallVector<mlir::Value>>
HugrTypeConverter::materializeTargetTuple(
    OpBuilder& rw, TypeRange ts, Value src, Location loc) const {
  auto tt = llvm::dyn_cast<TupleType>(src.getType());
  if (!tt) {
    return failure();
  }
  auto mb_partition = partitionTupleType(tt);
  if (failed(mb_partition) || mb_partition->getConvertedTypes() != ts) {
    return failure();
  }

  SmallVector<Value> unpack_results;
  rw.createOrFold<hugr_mlir::UnpackTupleOp>(unpack_results, loc, ts, src);
  SmallVector<Value> results;
  for (auto [i, v] : llvm::enumerate(unpack_results)) {
    if (this->isLegal(v.getType())) {
      results.push_back(v);
    } else if (
        auto mb_vs = this->materializeTargetConversion(
            rw, loc, mb_partition->getConvertedTypes(i), v)) {
      llvm::copy(*mb_vs, std::back_inserter(results));
    } else {
      return failure();
    }
  }
  return success(std::move(results));
}

mlir::FailureOr<mlir::Value> HugrTypeConverter::materializeSourceSum(
    OpBuilder& rw, hugr_mlir::SumType st, ValueRange vs, Location loc) const {
  if (st.numAlts() == 0) {
    return failure();
  }

  auto mb_partition = partitionSumType(st);
  if (failed(mb_partition)) {
    return failure();
  }

  SmallVector<Type> expected_tys{rw.getIndexType()};
  llvm::copy(
      mb_partition->getConvertedTypes(), std::back_inserter(expected_tys));
  if (expected_tys != vs.getTypes()) {
    return failure();
  }

  assert(vs.size() > 0 && "must have tag");
  auto tag = vs[0];
  auto converted_vs = vs.drop_front();

  SmallVector<int64_t> cases;
  for (auto i = 1; i < st.numAlts(); ++i) {
    cases.push_back(i);
  }
  auto switch_op =
      rw.create<scf::IndexSwitchOp>(loc, st, tag, cases, cases.size());
  OpBuilder::InsertionGuard _g(rw);

  for (auto [r, src_t] :
       llvm::zip_equal(switch_op.getRegions(), st.getTypes())) {
    assert(r->empty() && "Region should not have been constructed with block");
    auto n = r->getRegionNumber();
    rw.setInsertionPointToStart(rw.createBlock(r));

    Value v;
    if (!this->isLegal(src_t)) {
      v = materializeSourceConversion(
          rw, loc, src_t, mb_partition->getConvertedValues(converted_vs, n));
    } else {
      assert(
          mb_partition->getConvertedValues(converted_vs, n).getTypes() ==
              TypeRange{src_t} &&
          "must");
      v = mb_partition->getConvertedValues(converted_vs, n)[0];
    }
    assert(v && "can't fail");
    v = rw.createOrFold<hugr_mlir::TagOp>(loc, st, v, n);
    rw.create<scf::YieldOp>(loc, v);
  }
  return success(switch_op.getResult(0));
}

mlir::FailureOr<mlir::Value> HugrTypeConverter::materializeSourceTuple(
    OpBuilder& rw, TupleType tt, ValueRange vs, Location loc) const {
  auto mb_partition = partitionTupleType(tt);
  if (failed(mb_partition) ||
      mb_partition->getConvertedTypes() != vs.getTypes()) {
    return failure();
  }
  SmallVector<Value> tuple_args;
  for (auto i = 0; i < tt.size(); ++i) {
    auto t = tt.getType(i);
    auto i_vs = mb_partition->getConvertedValues(vs, i);
    if (isLegal(t)) {
      assert(i_vs.size() == 1 && "must");
      llvm::copy(i_vs, std::back_inserter(tuple_args));
    } else {
      auto v = materializeSourceConversion(rw, loc, t, i_vs);
      if (!v) {
        return failure();
      }
      tuple_args.push_back(v);
    }
  }
  return success(rw.createOrFold<hugr_mlir::MakeTupleOp>(loc, tt, tuple_args));
}

HugrTypeConverter::HugrTypeConverter() : OneToNTypeConverter() {
  addConversion([](Type t) -> std::optional<Type> {
    if (llvm::isa<hugr_mlir::SumType, hugr_mlir::FunctionType, TupleType>(t)) {
      return nullptr;
    }
    return {t};
  });
  addConversion([this](TupleType tt, llvm::SmallVectorImpl<Type>& result) -> std::optional<LogicalResult> {
    SmallVector<Type> ts;
    if (failed(this->convertTypes(tt.getTypes(), result))) {
      return std::nullopt;
    }
    return success();
  });
  addConversion([this](hugr_mlir::SumType st, llvm::SmallVectorImpl<Type>& result) -> std::optional<LogicalResult> {
    result.push_back(IndexType::get(st.getContext()));
    if (failed(this->convertTypes(st.getTypes(), result))) {
      result.clear();
      return std::nullopt;
    }
    return success();
  });
  // addConversion([this](hugr_mlir::ClosureType t) -> std::optional<Type> {
  //     return MemRefType::get({}, LLVM::LLVMStructType::getOpaque("_.hugr.closure", t.getContext()));
  // });
  addConversion([this](FunctionType t) -> std::optional<Type> {
    SmallVector<Type> inputs;
    if (failed(this->convertTypes(t.getInputs(), inputs))) {
      return std::nullopt;
    }
    SmallVector<Type> outputs;
    if (failed(this->convertTypes(t.getResults(), outputs))) {
      return std::nullopt;
    }
    return FunctionType::get(t.getContext(), inputs, outputs);
  });

  addSourceMaterialization(
      [this](
          OpBuilder& rw, TupleType tt, ValueRange vs,
          Location loc) -> std::optional<Value> {
        auto p = this->materializeSourceTuple(rw, tt, vs, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
  addTargetMaterialization(
      [this](
          OpBuilder& rw, TypeRange ts, Value src,
          Location loc) -> std::optional<SmallVector<Value>> {
        auto tt = llvm::dyn_cast<TupleType>(src.getType());
        if (!tt) {
          return std::nullopt;
        }
        auto p = this->materializeTargetTuple(rw, ts, src, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
  addArgumentMaterialization(
      [this](
          OpBuilder& rw, TupleType tt, ValueRange vs,
          Location loc) -> std::optional<Value> {
        auto p = this->materializeSourceTuple(rw, tt, vs, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });

  addSourceMaterialization(
      [this](
          OpBuilder& rw, hugr_mlir::SumType st, ValueRange vs,
          Location loc) -> std::optional<Value> {
        auto p = this->materializeSourceSum(rw, st, vs, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
  addTargetMaterialization(
      [this](
          OpBuilder& rw, TypeRange ts, Value src,
          Location loc) -> std::optional<SmallVector<Value>> {
        auto st = llvm::dyn_cast<hugr_mlir::SumType>(src.getType());
        if (!st) {
          return std::nullopt;
        }
        auto p = this->materializeTargetSum(rw, ts, src, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
  addArgumentMaterialization(
      [this](
          OpBuilder& rw, hugr_mlir::SumType st, ValueRange vs,
          Location loc) -> std::optional<Value> {
        auto p = this->materializeSourceSum(rw, st, vs, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
  // addSourceMaterialization([&](OpBuilder &builder, Type resultType,
  //                            ValueRange inputs,
  //                            Location loc) -> std::optional<Value> {
  //   if (inputs.size() != 1)
  //     return std::nullopt;

  //   return builder.create<UnrealizedConversionCastOp>(loc, resultType,
  //   inputs)
  //     .getResult(0);
  // });
  // addTargetMaterialization([&](OpBuilder &builder, TypeRange resultType,
  // Value inputs,
  //                            Location loc) -> std::optional<Value> {
  //   if (inputs.size() != 1)
  //     return std::nullopt;

  //   return builder.create<UnrealizedConversionCastOp>(loc, resultType,
  //   inputs)
  //     .getResult(0);
  // });
}

FailureOr<Value> SimpleHugrTypeConverter::materializeTargetSum(
    OpBuilder& rw, hugr_mlir::SumType dest_t, Value src, Location loc) const {
  assert(false && "unimplemented");
}

FailureOr<Value> SimpleHugrTypeConverter::materializeTargetTuple(
    OpBuilder& rw, TupleType dest_t, Value src, Location loc) const {
  SmallVector<Value> unpacked;
  rw.createOrFold<hugr_mlir::UnpackTupleOp>(
      unpacked, loc, dest_t.getTypes(), src);
  SmallVector<Value> converted;
  for (auto v : unpacked) {
    if (!isLegal(v.getType())) {
      v = materializeTargetConversion(rw, loc, convertType(v.getType()), v);
    }
    converted.push_back(v);
  }
  return success(
      rw.createOrFold<hugr_mlir::MakeTupleOp>(loc, dest_t, converted));
}

FailureOr<Value> SimpleHugrTypeConverter::materializeSourceSum(
    OpBuilder&, hugr_mlir::SumType, TypedValue<hugr_mlir::SumType>,
    Location) const {
  assert(false && "unimplemented");
}

FailureOr<Value> SimpleHugrTypeConverter::materializeSourceTuple(
    OpBuilder& rw, TupleType dest_t, TypedValue<TupleType> src,
    Location loc) const {
  if (dest_t.size() != src.getType().size()) {
    return failure();
  }
  SmallVector<Value> unpacked;
  rw.createOrFold<hugr_mlir::UnpackTupleOp>(
      unpacked, loc, src.getType().getTypes(), src);
  SmallVector<Value> unconverted;
  for (auto [dt, v] : llvm::zip_equal(dest_t.getTypes(), unpacked)) {
    if (dt != v.getType()) {
      v = materializeSourceConversion(rw, loc, dt, v);
    }
    unconverted.push_back(v);
  }
  return success(
      rw.createOrFold<hugr_mlir::MakeTupleOp>(loc, dest_t, unconverted));
}

SimpleHugrTypeConverter::SimpleHugrTypeConverter() {
  addConversion([](Type t) { return t; });
  addConversion([this](TupleType tt) -> std::optional<Type> {
    SmallVector<Type> ts;
    if (failed(this->convertTypes(tt.getTypes(), ts))) {
      return nullptr;
    }
    return TupleType::get(tt.getContext(), ts);
  });
  addConversion([this](hugr_mlir::SumType st) -> std::optional<Type> {
    SmallVector<Type> ts;
    if (failed(this->convertTypes(st.getTypes(), ts))) {
      return nullptr;
    }
    return hugr_mlir::SumType::get(st.getContext(), ts);
  });
  addSourceMaterialization(
      [this](
          OpBuilder& rw, TupleType dest_t, ValueRange vs,
          Location loc) -> std::optional<Value> {
        TypedValue<TupleType> src;
        if (vs.size() != 1 ||
            !(src = llvm::dyn_cast<TypedValue<TupleType>>(vs.front()))) {
          return std::nullopt;
        }
        auto p = this->materializeSourceTuple(rw, dest_t, src, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
  addTargetMaterialization(
      [this](
          OpBuilder& rw, TupleType dest_t, ValueRange vs,
          Location loc) -> std::optional<Value> {
        TypedValue<TupleType> src;
        if (vs.size() != 1 ||
            !(src = llvm::dyn_cast<TypedValue<TupleType>>(vs.front()))) {
          return std::nullopt;
        }
        auto p = this->materializeTargetTuple(rw, dest_t, src, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
  addArgumentMaterialization(
      [this](
          OpBuilder& rw, TupleType dest_t, ValueRange vs,
          Location loc) -> std::optional<Value> {
        TypedValue<TupleType> src;
        if (vs.size() != 1 ||
            !(src = llvm::dyn_cast<TypedValue<TupleType>>(vs.front()))) {
          return std::nullopt;
        }
        auto p = this->materializeSourceTuple(rw, dest_t, src, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });

  addSourceMaterialization(
      [this](
          OpBuilder& rw, hugr_mlir::SumType dest_t, ValueRange vs,
          Location loc) -> std::optional<Value> {
        TypedValue<hugr_mlir::SumType> src;
        if (vs.size() != 1 ||
            !(src =
                  llvm::dyn_cast<TypedValue<hugr_mlir::SumType>>(vs.front()))) {
          return std::nullopt;
        }
        auto p = this->materializeSourceSum(rw, dest_t, src, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
  addTargetMaterialization(
      [this](
          OpBuilder& rw, hugr_mlir::SumType dest_t, ValueRange vs,
          Location loc) -> std::optional<Value> {
        TypedValue<hugr_mlir::SumType> src;
        if (vs.size() != 1 ||
            !(src =
                  llvm::dyn_cast<TypedValue<hugr_mlir::SumType>>(vs.front()))) {
          return std::nullopt;
        }
        auto p = this->materializeTargetSum(rw, dest_t, src, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
  addArgumentMaterialization(
      [this](
          OpBuilder& rw, hugr_mlir::SumType dest_t, ValueRange vs,
          Location loc) -> std::optional<Value> {
        TypedValue<hugr_mlir::SumType> src;
        if (vs.size() != 1 ||
            !(src =
                  llvm::dyn_cast<TypedValue<hugr_mlir::SumType>>(vs.front()))) {
          return std::nullopt;
        }
        auto p = this->materializeSourceSum(rw, dest_t, src, loc);
        if (failed(p)) {
          return std::nullopt;
        }
        return *p;
      });
}
std::unique_ptr<mlir::TypeConverter> hugr_mlir::createTypeConverter() {
  return std::make_unique<HugrTypeConverter>();
}

std::unique_ptr<mlir::TypeConverter> hugr_mlir::createSimpleTypeConverter() {
  return std::make_unique<SimpleHugrTypeConverter>();
}
