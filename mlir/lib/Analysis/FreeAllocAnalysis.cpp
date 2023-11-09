#include "hugr-mlir/Analysis/FreeAllocAnalysis.h"


void hugr_mlir::FreeAllocForwardAnalyis::visitOperation(mlir::Operation* op,  mlir::ArrayRef<const FreeAllocLattice *> operands,
                                                 mlir::ArrayRef<FreeAllocLattice *> results) {
    auto mei = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op);
    if(!mei) { return; }

    for(auto r: results) {
        llvm::SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> effects;
        mei.getEffectsOnValue(r->getPoint(), effects);
        for(auto e: llvm::make_filter_range(effects, [=](auto x) { return x.getResource() == resource; })) {
        }
    }
}

void hugr_mlir::FreeAllocForwardAnalyis::setToEntryState(FreeAllocLattice *lattice) {

}

void hugr_mlir::FreeAllocBackwardAnalyis::visitOperation(mlir::Operation *op, mlir::ArrayRef<FreeAllocLattice *> operands,
                                                          mlir::ArrayRef<const FreeAllocLattice *> results) {

}

void hugr_mlir::FreeAllocBackwardAnalyis::setToExitState(FreeAllocLattice *lattice) {
}
