#include "hugr-mlir/IR/HugrTypes.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrTypeInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "hugr-mlir/IR/HugrTypes.cpp.inc"

auto hugr_mlir::FunctionType::clone(
    mlir::TypeRange new_args, mlir::TypeRange new_results) -> FunctionType {
  return get(
      getContext(), getExtensions(),
      getFunctionType().clone(new_args, new_results));
}

mlir::LogicalResult hugr_mlir::FunctionType::verify(
    ::llvm::function_ref< ::mlir::InFlightDiagnostic()>,
    hugr_mlir::ExtensionSetAttr, mlir::FunctionType) {
  return mlir::success();
}

hugr_mlir::ExtendedType hugr_mlir::ExtendedType::removeExtensions(
    ExtensionSetAttr to_remove) {
  return ExtendedType::get(
      getContext(), getExtensions().remove(to_remove), getInnerType(),
      getConstraint());
}

bool hugr_mlir::areTypesCompatible(mlir::Type lhs, mlir::Type rhs) {
  if (lhs == rhs) {
    return true;
  }
  auto go = [](auto l, auto r) -> bool {
    return llvm::TypeSwitch<mlir::Type, bool>(l)
        .Case([=](ExtendedType t) {
          return t.getExtensions().size() == 0 &&
                 areTypesCompatible(t.getInnerType(), r);
        })
        .Default(false);
  };
  return go(lhs, rhs) || go(rhs, lhs);
}

bool hugr_mlir::areTypesCompatible(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip_equal(lhs, rhs), [](auto x) {
           auto [l, r] = x;
           return areTypesCompatible(l, r);
         });
}

void hugr_mlir::HugrDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hugr-mlir/IR/HugrTypes.cpp.inc"
      >();
}
