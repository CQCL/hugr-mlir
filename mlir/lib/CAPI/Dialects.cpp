#include "hugr-mlir-c/Dialects.h"

#include "hugr-mlir/IR/HugrAttrs.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrEnums.h"
#include "hugr-mlir/IR/HugrTypes.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Hugr, hugr, hugr_mlir::HugrDialect)

using namespace mlir;

MlirAttribute mlirHugrTypeConstraintAttrGet(
    MlirContext context, const char* kind) {
  MLIRContext* ctx(unwrap(context));
  if (auto x = hugr_mlir::symbolizeTypeConstraint(kind)) {
    return wrap(hugr_mlir::TypeConstraintAttr::get(ctx, *x));
  }
  return {nullptr};
}

MlirType mlirHugrSumTypeGet(
    MlirContext context, int32_t n, MlirType* components) {
  llvm::SmallVector<Type> components_unwrapped;
  std::transform(
      components, components + n, std::back_inserter(components_unwrapped),
      [](auto x) { return unwrap(x); });
  return wrap(hugr_mlir::SumType::get(unwrap(context), components_unwrapped));
}
