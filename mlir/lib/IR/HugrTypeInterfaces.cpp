// clang-format off
#include "hugr-mlir/IR/HugrTypeInterfaces.h"
#include "hugr-mlir/IR/HugrTypeInterfaces.cpp.inc"
//clang-format on

#include "hugr-mlir/IR/HugrAttrs.h"
#include "hugr-mlir/IR/HugrTypes.h"
#include "hugr-mlir/IR/HugrDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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
    auto acc = ExtensionSetAttr::get(t.getContext());
    for(auto t: llvm::cast<mlir::TupleType>(t).getTypes()) {
      if(auto hti = llvm::dyn_cast<HugrTypeInterface>(t)) {
        acc = acc.merge(hti.getExtensions());
      }
    }
    return acc;
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    auto acc = TypeConstraintAttr::get(t.getContext(), TypeConstraint::Copyable);
    for(auto t: llvm::cast<mlir::TupleType>(t).getTypes()) {
      if(auto hti = llvm::dyn_cast<HugrTypeInterface>(t)) {
        acc = acc.intersection(hti.getConstraint());
      }
    }
    return acc.getValue();
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
    auto acc = ExtensionSetAttr::get(t.getContext());
    for(auto t: llvm::cast<SumType>(t).getTypes()) {
      if(auto hti = llvm::dyn_cast<HugrTypeInterface>(t)) {
        acc = acc.merge(hti.getExtensions());
      }
    }
    return acc;
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    auto acc = TypeConstraintAttr::get(t.getContext(), TypeConstraint::Copyable);
    for(auto t: llvm::cast<SumType>(t).getTypes()) {
      if(auto hti = llvm::dyn_cast<HugrTypeInterface>(t)) {
        acc = acc.intersection(hti.getConstraint());
      }
    }
    return acc.getValue();
  }
};

struct OpaqueHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          OpaqueHugrTypeInterfaceModel, OpaqueType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return llvm::cast<OpaqueType>(t).getConstraint();
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

struct HugrFunctionHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          HugrFunctionHugrTypeInterfaceModel, FunctionType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Copyable;
  }
};

struct ExternalFunctionHugrTypeInterfaceModel
    : public HugrTypeInterface::ExternalModel<
          ExternalFunctionHugrTypeInterfaceModel, mlir::FunctionType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Equatable;
  }
};

struct ExternalMemRefHugrTypeInterfaceModule
    : public HugrTypeInterface::ExternalModel<
          ExternalMemRefHugrTypeInterfaceModule, mlir::MemRefType> {
  ExtensionSetAttr getExtensions(mlir::Type t) const {
    return ExtensionSetAttr::get(t.getContext());
  }
  TypeConstraint getConstraint(mlir::Type t) const {
    return TypeConstraint::Equatable;
  }
};


}  // namespace

void hugr_mlir::HugrDialect::registerTypeInterfaces() {
  auto& context = *getContext();
  SumType::attachInterface<SumHugrTypeInterfaceModel>(context);
  OpaqueType::attachInterface<OpaqueHugrTypeInterfaceModel>(context);
  AliasRefType::attachInterface<AliasRefHugrTypeInterfaceModel>(context);
  ExtendedType::attachInterface<ExtendedHugrTypeInterfaceModel>(context);
  FunctionType::attachInterface<HugrFunctionHugrTypeInterfaceModel>(context);

  mlir::IndexType::attachInterface<ExternalIndexHugrTypeInterfaceModel>(
      context);
  mlir::TupleType::attachInterface<ExternalTupleHugrTypeInterfaceModel>(
      context);
  mlir::Float32Type::attachInterface<ExternalFloat32HugrTypeInterfaceModel>(
      context);
  mlir::Float64Type::attachInterface<ExternalFloat64HugrTypeInterfaceModel>(
      context);
  mlir::IntegerType::attachInterface<ExternalIntegerHugrTypeInterfaceModel>(
      context);
  mlir::FunctionType::attachInterface<ExternalFunctionHugrTypeInterfaceModel>(context);

  mlir::MemRefType::attachInterface<ExternalMemRefHugrTypeInterfaceModule>(context);
}
