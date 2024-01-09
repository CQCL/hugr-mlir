#include "hugr-mlir/IR/HugrAttrs.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "hugr-mlir/IR/HugrAttrs.cpp.inc"

mlir::FailureOr<mlir::SmallVector<hugr_mlir::ExtensionAttr>>
hugr_mlir::ExtensionSetAttr::parseExtensionList(mlir::AsmParser& parser) {
  std::string s;
  mlir::SmallVector<ExtensionAttr> result;
  if (!parser.parseOptionalString(&s)) {
    result.push_back(parser.getBuilder().getAttr<ExtensionAttr>(s));

    for (;;) {
      if (parser.parseOptionalComma()) {
        break;
      }
      if (parser.parseString(&s)) {
        return mlir::failure();
      }
      result.push_back(parser.getBuilder().getAttr<ExtensionAttr>(s));
    }
  }
  return mlir::success(std::move(result));
}

mlir::LogicalResult hugr_mlir::StaticEdgeAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type, mlir::SymbolRefAttr ref) {
  if (!type || llvm::isa<mlir::NoneType>(type)) {
    return emitError() << "Invalid type: " << type;
  }
  return mlir::success();
}

mlir::TupleType hugr_mlir::UnitAttr::getType() {
  return mlir::TupleType::get(getContext());
}

hugr_mlir::TypeConstraintAttr hugr_mlir::TypeConstraintAttr::intersection(
    TypeConstraintAttr other) {
  return intersection(other.getValue());
}

hugr_mlir::TypeConstraintAttr hugr_mlir::TypeConstraintAttr::intersection(
    TypeConstraint other) {
  return TypeConstraintAttr::get(getContext(), std::min(getValue(), other));
}

hugr_mlir::ExtensionSetAttr hugr_mlir::ExtensionSetAttr::remove(
    ExtensionSetAttr rhs) {
  mlir::DenseSet<ExtensionAttr> rhs_set;
  for (auto x : rhs.getExtensions()) {
    rhs_set.insert(x);
  };
  mlir::SmallVector<ExtensionAttr> new_attrs;
  llvm::copy_if(
      getExtensions(), std::back_inserter(new_attrs),
      [&rhs_set](auto x) { return !rhs_set.contains(x); });
  return ExtensionSetAttr::get(getContext(), new_attrs);
}

hugr_mlir::ExtensionSetAttr hugr_mlir::ExtensionSetAttr::merge(
    ExtensionSetAttr rhs) {
  llvm::SmallVector<ExtensionAttr> all{getExtensions()};
  llvm::copy(rhs.getExtensions(), std::back_inserter(all));
  return ExtensionSetAttr::get(getContext(), all);
}

auto hugr_mlir::SumAttr::getSumType() -> SumType {
  return llvm::cast<SumType>(getType());
}

mlir::IntegerAttr hugr_mlir::SumAttr::getTagAttr() {
  return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), getTag());
}

mlir::TupleType hugr_mlir::TupleAttr::getType() {
  mlir::SmallVector<mlir::Type> ts;
  llvm::transform(
      getValues(), std::back_inserter(ts), [](auto x) { return x.getType(); });
  return mlir::TupleType::get(getContext(), ts);
}

void hugr_mlir::HugrDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hugr-mlir/IR/HugrAttrs.cpp.inc"
      >();
}
