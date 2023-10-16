// clang-format off
#include "hugr-mlir/IR/HugrTypeInterfaces.h"
#include "hugr-mlir/IR/HugrTypeInterfaces.cpp.inc"
//clang-format on

#include "hugr-mlir/IR/HugrAttrs.h"
#include "hugr-mlir/IR/HugrTypes.h"
#include "hugr-mlir/IR/HugrDialect.h"

namespace {
using namespace hugr_mlir;

struct ExternalIndexHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          ExternalIndexHugrTypeInterfaceModel, mlir::IndexType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Equatable;
  }
};

struct ExternalTupleHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          ExternalTupleHugrTypeInterfaceModel, mlir::TupleType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Copyable;
  }
};

struct ExternalIntegerHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          ExternalIntegerHugrTypeInterfaceModel, mlir::IntegerType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Equatable;
  }
};

struct ExternalFloat32HugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          ExternalFloat32HugrTypeInterfaceModel, mlir::Float32Type> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Copyable;
  }
};

struct ExternalFloat64HugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          ExternalFloat64HugrTypeInterfaceModel, mlir::Float64Type> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Copyable;
  }
};

struct SumHugrTypeInterfaceModel : public HugrTypeInterface::ExternalModel<
                                       SumHugrTypeInterfaceModel, SumType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Copyable;
  }
};

struct OpaqueHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          OpaqueHugrTypeInterfaceModel, OpaqueType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return llvm::cast<AliasRefType>(t).getConstraint();
  }
};

struct AliasRefHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          AliasRefHugrTypeInterfaceModel, AliasRefType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return llvm::cast<AliasRefType>(t).getExtensions();
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return llvm::cast<AliasRefType>(t).getConstraint();
  }
};

struct ExtendedHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          ExtendedHugrTypeInterfaceModel, ExtendedType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return llvm::cast<ExtendedType>(t).getExtensions();
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return llvm::cast<ExtendedType>(t).getConstraint();
  }
};

struct FunctionHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          ExtendedHugrTypeInterfaceModel, FunctionType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Copyable;
  }
};

}  // namespace

void hugr_mlir::HugrDialect::registerTypeInterfaces() {
  SumType::attachInterface<SumHugrTypeInterfaceModel>(*getContext());
  OpaqueType::attachInterface<OpaqueHugrTypeInterfaceModel>(*getContext());
  AliasRefType::attachInterface<AliasRefHugrTypeInterfaceModel>(*getContext());
  ExtendedType::attachInterface<ExtendedHugrTypeInterfaceModel>(*getContext());
  FunctionType::attachInterface<FunctionHugrTypeInterfaceModel>(*getContext());

  mlir::IndexType::attachInterface<ExternalIndexHugrTypeInterfaceModel>(
      *getContext());
  mlir::TupleType::attachInterface<ExternalTupleHugrTypeInterfaceModel>(
      *getContext());
  mlir::Float32Type::attachInterface<ExternalFloat32HugrTypeInterfaceModel>(
      *getContext());
  mlir::Float64Type::attachInterface<ExternalFloat64HugrTypeInterfaceModel>(
      *getContext());
  mlir::IntegerType::attachInterface<ExternalIntegerHugrTypeInterfaceModel>(
      *getContext());
}
