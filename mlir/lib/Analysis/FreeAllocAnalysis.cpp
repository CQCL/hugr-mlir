#include "hugr-mlir/Analysis/FreeAllocAnalysis.h"

#include "hugr-mlir/IR/HugrOps.h"
#include "hugr-mlir/IR/HugrTypeInterfaces.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"

#include "llvm/ADT/TypeSwitch.h"

mlir::ChangeResult hugr_mlir::join_alloc_state(AllocState& lhs, AllocState rhs) {
    if(lhs == rhs) { return mlir::ChangeResult::NoChange; }
    lhs = AllocState::Unknown;
    return mlir::ChangeResult::Change;
}

mlir::ChangeResult hugr_mlir::FreeAllocState::observe(mlir::Value k, AllocState v) {
    auto [it, worked] = alloc_map.try_emplace(k, v);
    return worked ? mlir::ChangeResult::Change : join_alloc_state(it->second, v);
}

void hugr_mlir::FreeAllocAnalysisState::print(llvm::raw_ostream &os) const {

}

void hugr_mlir::FreeAllocDenseAnalysis::setToEntryState(LatticeT *lattice) {
    auto r = mlir::ChangeResult::NoChange;
    auto b = llvm::cast<mlir::Block*>(lattice->getPoint());
    for(auto lv: linear_values_range(b->getArguments())) {
        r |= lattice->state.observe(lv, AllocState::Allocated);
    }
    propagateIfChanged(lattice, r);
}

void hugr_mlir::FreeAllocDenseAnalysis::visitOperation(mlir::Operation *op, const LatticeT &before,
                                                       LatticeT *after) {
    LatticeT l(before);
    if(auto mo = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        llvm::SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> effects;
        mo.getEffectsOnResource(hugr_mlir::LinearityResource::get(), effects);
        for(auto const& e: effects) {
            auto v = e.getValue();
            if(!v) { continue; }
            assert(llvm::isa<LinearValue>(v) && "only linear values should have linear effects");
            if(llvm::isa<mlir::MemoryEffects::Allocate>(*e.getEffect())) {
                l.state.observe(v, AllocState::Allocated);
            } else if (llvm::isa<mlir::MemoryEffects::Free>(*e.getEffect())) {
                l.state.observe(v, AllocState::Free);
            } else {
                    // TODO assert?
            };
        }
    }
    join(after, l);
}

// void hugr_mlir::FreeAllocForwardAnalyis::visitOperation(mlir::Operation* op,  mlir::ArrayRef<const FreeAllocLattice *> operands,
//                                                  mlir::ArrayRef<FreeAllocLattice *> results) {
//     auto mei = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op);
//     if(!mei) { return; }

//     for(auto r: results) {
//         llvm::SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> effects;
//         mei.getEffectsOnValue(r->getPoint(), effects);
//         for(auto e: llvm::make_filter_range(effects, [=](auto x) { return x.getResource() == resource; })) {
//         }
//     }
// }

// void hugr_mlir::FreeAllocForwardAnalyis::setToEntryState(FreeAllocLattice *lattice) {

// }

// void hugr_mlir::FreeAllocBackwardAnalyis::visitOperation(mlir::Operation *op, mlir::ArrayRef<FreeAllocLattice *> operands,
//                                                           mlir::ArrayRef<const FreeAllocLattice *> results) {

// }

// void hugr_mlir::FreeAllocBackwardAnalyis::setToExitState(FreeAllocLattice *lattice) {
// }

mlir::LogicalResult hugr_mlir::FreeAllocAnalysis::initialise() {
    if(solver) { return mlir::success(); }

    solver.emplace();
    solver->load<mlir::dataflow::SparseConstantPropagation>();
    solver->load<mlir::dataflow::DeadCodeAnalysis>();
    solver->load<FreeAllocDenseAnalysis>();
    auto r = solver->initializeAndRun(top);
    if(mlir::failed(r)) {
        solver = std::nullopt;
    }
    return r;
}
