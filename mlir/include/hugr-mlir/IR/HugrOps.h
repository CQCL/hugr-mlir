#ifndef HUGR_MLIR_IR_HUGR_OPS_H
#define HUGR_MLIR_IR_HUGR_OPS_H

// clang-format off
#include "hugr-mlir/IR/HugrTypes.h"
#include "hugr-mlir/IR/HugrTypeInterfaces.h"


#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
// clang-format on

namespace hugr_mlir {

void getHugrTypeMemoryEffects(
    mlir::Operation *,
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> &);
bool isDataflowGraphRegion(mlir::Region &);
bool isControlFlowGraphRegion(mlir::Region &);

}  // namespace hugr_mlir

namespace mlir::OpTrait {

template <typename ConcreteOp>
struct HugrTypeMemoryEffectsTrait
    : TraitBase<ConcreteOp, HugrTypeMemoryEffectsTrait> {
  void getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>> &effects) {
    return hugr_mlir::getHugrTypeMemoryEffects(this->getOperation(), effects);
  }
};

}  // namespace mlir::OpTrait

#define GET_OP_CLASSES
#include "hugr-mlir/IR/HugrOps.h.inc"

namespace hugr_mlir {

struct LinearityResource
    : mlir::SideEffects::Resource::Base<LinearityResource> {
  mlir::StringRef getName() final { return "Linearity"; }
};

mlir::ParseResult parseCallInputsOutputs(
    ::mlir::OpAsmParser &, StaticEdgeAttr &,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &,
    mlir::SmallVectorImpl<mlir::Type> &, mlir::SmallVectorImpl<mlir::Type> &);
void printCallInputsOutputs(
    ::mlir::OpAsmPrinter &, CallOp, StaticEdgeAttr, mlir::OperandRange,
    mlir::TypeRange, mlir::TypeRange);

mlir::ParseResult parseStaticEdge(::mlir::OpAsmParser &, StaticEdgeAttr &);
void printStaticEdge(::mlir::OpAsmPrinter &, mlir::Operation *, StaticEdgeAttr);

mlir::ParseResult parseTailLoopOpOutputTypes(
    ::mlir::OpAsmParser &, mlir::SmallVectorImpl<mlir::Type> const &,
    mlir::SmallVectorImpl<mlir::Type> &, mlir::SmallVectorImpl<mlir::Type> &);
void printTailLoopOpOutputTypes(
    ::mlir::OpAsmPrinter &, mlir::Operation *, mlir::TypeRange, mlir::TypeRange,
    mlir::TypeRange);

}  // namespace hugr_mlir

#endif  // HUGR_MLIR_IR_HUGR_OPS_H