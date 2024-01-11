#include "hugr-mlir-c/Dialects.h"

#include "hugr-mlir/IR/HugrAttrs.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrEnums.h"
#include "hugr-mlir/IR/HugrTypes.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Hugr, hugr, hugr_mlir::HugrDialect)

using namespace mlir;

MlirAttribute mlirHugrTypeConstraintAttrGet(
    MlirContext context, MlirStringRef kind) {
  if (auto x = hugr_mlir::symbolizeTypeConstraint(unwrap(kind))) {
    return wrap(hugr_mlir::TypeConstraintAttr::get(unwrap(context), *x));
  }
  return {nullptr};
}

MlirType mlirHugrSumTypeGet(
    MlirContext context, int32_t n, MlirType const* components) {
  llvm::SmallVector<Type> components_unwrapped;
  std::transform(
      components, components + n, std::back_inserter(components_unwrapped),
      [](auto x) { return unwrap(x); });
  return wrap(hugr_mlir::SumType::get(unwrap(context), components_unwrapped));
}

bool mlirTypeIsAHugrFunctionType(MlirType t) {
  return llvm::isa_and_present<hugr_mlir::FunctionType>(unwrap(t));
}

MlirType mlirHugrFunctionTypeGet(
    MlirAttribute extensions, MlirType function_type) {
  auto ft = llvm::cast<FunctionType>(unwrap(function_type));
  auto es = llvm::cast<hugr_mlir::ExtensionSetAttr>(unwrap(extensions));
  return wrap(hugr_mlir::FunctionType::get(es, ft));
}

bool mlirAttributeIsAHugrExtensionAttr(MlirAttribute a) {
  return llvm::isa_and_present<hugr_mlir::ExtensionAttr>(unwrap(a));
}

MlirAttribute mlirHugrExtensionAttrGet(MlirContext context, MlirStringRef ext) {
  auto ctx = unwrap(context);
  return wrap(hugr_mlir::ExtensionAttr::get(ctx, unwrap(ext)));
}

bool mlirAttributeIsAHugrExtensionSetAttr(MlirAttribute a) {
  return llvm::isa_and_present<hugr_mlir::ExtensionSetAttr>(unwrap(a));
}

MlirAttribute mlirHugrExtensionSetAttrGet(
    MlirContext context, int32_t n_extensions,
    MlirAttribute const* extensions) {
  auto ctx = unwrap(context);
  llvm::SmallVector<hugr_mlir::ExtensionAttr> es;
  std::transform(
      extensions, extensions + n_extensions, std::back_inserter(es),
      [](auto x) { return llvm::cast<hugr_mlir::ExtensionAttr>(unwrap(x)); });
  return wrap(hugr_mlir::ExtensionSetAttr::get(ctx, es));
}

bool mlirTypeIsAHugrAliasRefType(MlirType t) {
  return llvm::isa_and_present<hugr_mlir::AliasRefType>(unwrap(t));
}

MlirType mlirHugrAliasRefTypeGet(
    MlirAttribute extensions, MlirAttribute sym_ref,
    MlirAttribute type_constraint) {
  auto es = llvm::cast<hugr_mlir::ExtensionSetAttr>(unwrap(extensions));
  auto sym = llvm::cast<SymbolRefAttr>(unwrap(sym_ref));
  auto constraint_attr =
      llvm::cast<hugr_mlir::TypeConstraintAttr>(unwrap(type_constraint));
  return wrap(
      hugr_mlir::AliasRefType::get(es, sym, constraint_attr.getValue()));
}

bool mlirTypeIsAHugrOpaqueType(MlirType t) {
  return llvm::isa_and_present<hugr_mlir::OpaqueType>(unwrap(t));
}

MlirType mlirHugrOpaqueTypeGet(
    MlirStringRef name, MlirAttribute extension, MlirAttribute type_constraint,
    intptr_t n_args, MlirAttribute const* args) {
  llvm::SmallVector<Attribute> as;
  std::transform(args, args + n_args, std::back_inserter(as), [](auto x) {
    return unwrap(x);
  });
  auto e = llvm::cast<hugr_mlir::ExtensionAttr>(unwrap(extension));
  auto tc = llvm::cast<hugr_mlir::TypeConstraintAttr>(unwrap(type_constraint));
  return wrap(hugr_mlir::OpaqueType::get(unwrap(name), e, as, tc.getValue()));
}

bool mlirAttributeIsHugrStaticEdgeAttr(MlirAttribute attr) {
  return llvm::isa_and_present<hugr_mlir::StaticEdgeAttr>(unwrap(attr));
}

MlirAttribute mlirHugrStaticEdgeAttrGet(MlirType type, MlirAttribute sym) {
  return wrap(hugr_mlir::StaticEdgeAttr::get(
      unwrap(type), llvm::cast<SymbolRefAttr>(unwrap(sym))));
}

bool mlirAttributeIsHugrSumAttr(MlirAttribute x) {
  return llvm::isa_and_present<hugr_mlir::SumAttr>(unwrap(x));
}

MlirAttribute mlirHugrSumAttrGet(
    MlirType type, uint32_t tag, MlirAttribute value) {
  auto t = llvm::cast<hugr_mlir::SumType>(unwrap(type));
  return wrap(hugr_mlir::SumAttr::get(t, tag, llvm::cast<TypedAttr>(unwrap(value))));
}

MlirAttribute
mlirHugrTupleAttrGet(MlirContext context, intptr_t n, MlirAttribute const* value) {
  SmallVector<TypedAttr> attrs;
  for(auto i = 0u; i < n; ++i) { attrs.push_back(cast<TypedAttr>(unwrap(value[i]))); }
  return wrap(hugr_mlir::TupleAttr::get(unwrap(context), attrs));
}

bool mlirAttributeIsHugrTupleAttr(MlirAttribute a) {
  return mlir::isa<hugr_mlir::TupleAttr>(unwrap(a));
}
