#ifndef HUGR_MLIR_ANALYSIS_FREEALLOCANALYSIS_H
#define HUGR_MLIR_ANALYSIS_FREEALLOCANALYSIS_H

#include <variant>

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace hugr_mlir {

enum class AllocState { Allocated, Free, Unknown };

mlir::ChangeResult join_alloc_state(AllocState &lhs, AllocState rhs);

struct FreeAllocState {
  mlir::ChangeResult observe(mlir::Value, AllocState);
  mlir::ChangeResult join(FreeAllocState const &other) {
    auto r = mlir::ChangeResult::NoChange;
    for (auto const &[k, v] : other.alloc_map) {
      r |= observe(k, v);
    }
    return r;
  }

 private:
  llvm::DenseMap<mlir::Value, AllocState> alloc_map;
};

namespace detail {
struct top_t {
  bool operator==(top_t const &) const { return true; }
};
struct bottom_t {
  bool operator==(bottom_t const &) const { return true; }
};

// struct allocated_t {
//     bool operator==(allocated_t const&) const { return true; }
// };
// struct unallocated_t {
//     bool operator==(unallocated_t const&) const { return true; }
// };
// using variant = std::variant<top_t,bottom_t,allocated_t,unallocated_t>;
}  // namespace detail

struct FreeAllocAnalysisState : mlir::dataflow::AbstractDenseLattice {
  using AbstractDenseLattice::AbstractDenseLattice;
  void print(llvm::raw_ostream &os) const override;
  FreeAllocState state;
};
// FreeAllocState(variant_t t) : variant_t(t) {}
// FreeAllocState() : FreeAllocState(detail::top_t()) {}
// static FreeAllocState top() { return FreeAllocState(detail::top_t()); }
// static FreeAllocState bottom() { return FreeAllocState(detail::top_t()); }
// static FreeAllocState allocated() { return FreeAllocState(detail::top_t()); }
// static FreeAllocState unallocated() { return FreeAllocState(detail::top_t());
// }

// bool isTop() const { return *this == top(); }
// bool isBottom() const { return *this == top(); }

// bool operator==(FreeAllocState const& other) const {
//     return static_cast<variant_t const&>(*this) == static_cast<variant_t
//     const&>(other);
// }

// bool operator!=(FreeAllocState const& other) const {
//     return !(*this == other);
// }

// static FreeAllocState join(FreeAllocState const& lhs, FreeAllocState const&
// rhs) {
//     if(lhs.isTop() || rhs.isBottom()) { return rhs; }
//     if(rhs.isTop() || lhs.isBottom()) { return lhs; }
//     if(lhs != rhs) { return bottom(); }
//     return lhs;
// }

// void allocate() {
//     if (isTop() || *this == unallocated()) {
//         *this = allocated();
//     } else {
//         *this = bottom();
//     }
// }

// void unallocate() {
//     if (isTop() || *this == allocated()) {
//         *this = unallocated();
//     } else {
//         *this = bottom();
//     }
// }

// void print(llvm::raw_ostream& os) const {
//     if(isTop()) {
//         os << "<UNINITIALISED>";
//     } else if (isBottom()) {
//         os << "<BOTTOM>";
//     } else if (*this == allocated()) {
//         os << "allocated";
//     } else if (*this == unallocated()) {
//         os << "allocated";
//     } else {
//         assert(false && "must be one of those" );
//     }
// }

struct FreeAllocDenseAnalysis
    : mlir::dataflow::DenseForwardDataFlowAnalysis<FreeAllocAnalysisState> {
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;
  using LatticeT = FreeAllocAnalysisState;
  void setToEntryState(LatticeT *lattice) override;
  void visitOperation(
      mlir::Operation *op, const LatticeT &before, LatticeT *after) override;
};

struct FreeAllocAnalysis {
  FreeAllocAnalysis(mlir::Operation *top_) : top(top_) {}
  ~FreeAllocAnalysis() = default;
  mlir::LogicalResult initialise();

 protected:
 private:
  mlir::Operation *top;
  std::optional<mlir::DataFlowSolver> solver;
};
// using FreeAllocLattice = mlir::dataflow::Lattice<FreeAllocState>;

// struct FreeAllocForwardAnalyis :
// mlir::dataflow::SparseForwardDataFlowAnalysis<FreeAllocLattice> {
//   FreeAllocForwardAnalyis(mlir::DataFlowSolver& solver,
//   ::mlir::SideEffects::Resource const* _resource) :
//   SparseForwardDataFlowAnalysis(solver), resource(_resource) {} void
//   visitOperation(mlir::Operation *op, mlir::ArrayRef<const FreeAllocLattice
//   *> operands,
//                               mlir::ArrayRef<FreeAllocLattice *> results)
//                               override;
// protected:
//   void setToEntryState(FreeAllocLattice *lattice) override;
// private:
//   ::mlir::SideEffects::Resource const* resource;

// };

// struct FreeAllocBackwardAnalyis :
// mlir::dataflow::SparseBackwardDataFlowAnalysis<FreeAllocLattice> {
//   FreeAllocBackwardAnalyis(mlir::DataFlowSolver& solver,
//   mlir::SymbolTableCollection stc, ::mlir::SideEffects::Resource const*
//   _resource) : SparseBackwardDataFlowAnalysis(solver, stc),
//   resource(_resource) {}

//   void visitOperation(mlir::Operation *op, mlir::ArrayRef<FreeAllocLattice *>
//   operands,
//                               mlir::ArrayRef<const FreeAllocLattice *>
//                               results) override;
// protected:
//   void setToExitState(FreeAllocLattice *lattice) override;
// private:
//   ::mlir::SideEffects::Resource const* resource;
// };

}  // namespace hugr_mlir

#endif  // HUGR_MLIR_ANALYSIS_FREEALLOCANALYSIS_H
