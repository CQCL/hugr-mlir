// clang-format off
#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Interfaces/FoldInterfaces.h"

// clang-format on

#include "hugr-mlir/IR/HugrDialect.cpp.inc"

namespace {
struct HugrDialectFoldInterface : mlir::DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;
  bool shouldMaterializeInto(mlir::Region* r) const override {
    return llvm::isa<hugr_mlir::FuncOp>(r->getParentOp());
  }
};

}  // namespace

mlir::Operation* hugr_mlir::HugrDialect::materializeConstant(
    ::mlir::OpBuilder& rw, ::mlir::Attribute value, ::mlir::Type type,
    ::mlir::Location loc) {
  return mlir::TypeSwitch<mlir::Attribute, mlir::Operation*>(value)
      .Case([&](hugr_mlir::SumAttr a) -> mlir::Operation* {
        if (a.getType() != type && "must") {
          mlir::emitError(loc)
              << "falied to materializeConstant:" << value << " into " << type;
          return nullptr;
        }
        return rw.create<hugr_mlir::ConstantOp>(loc, a);
      })
      .Case([&](hugr_mlir::TupleAttr a) {
        assert(a.getType() == type && "must");
        return rw.create<hugr_mlir::ConstantOp>(loc, a);
      })
      .Default([](auto) { return nullptr; });
}

void hugr_mlir::HugrDialect::initialize() {
  registerTypes();
  registerAttrs();
  registerOps();
  registerTypeInterfaces();

  addInterfaces<HugrDialectFoldInterface>();
}
