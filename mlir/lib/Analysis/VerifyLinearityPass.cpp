#include "hugr-mlir/Analysis/VerifyLinearityPass.h"

#include "hugr-mlir/IR/HugrTypeInterfaces.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

namespace hugr_mlir {
#define GEN_PASS_DEF_HUGRVERIFYLINEARITYPASS
#include "hugr-mlir/Analysis/Passes.h.inc"
}  // namespace hugr_mlir

namespace {
using namespace mlir;

struct AllPredecessors : AnalysisState {
  using AnalysisState::AnalysisState;
  using BlockSet_t = hugr_mlir::VerifyLinearityAnalysis::BlockSet_t;
  void print(llvm::raw_ostream& os) const override;
  ChangeResult join(std::optional<DenseSet<Block*>> const&);
  ChangeResult join(AllPredecessors const* rhs) { return join(rhs->blocks); }
  BlockSet_t const& get() const { return blocks; }

 private:
  BlockSet_t blocks{{DenseSet<Block*>()}};
};

struct AllPredecessorsAnalysis : mlir::DataFlowAnalysis {
  using DataFlowAnalysis::DataFlowAnalysis;
  LogicalResult initialize(Operation* top) override;
  LogicalResult visit(ProgramPoint point) override;
};

struct HugrVerifyLinearityPass
    : hugr_mlir::impl::HugrVerifyLinearityPassBase<HugrVerifyLinearityPass> {
  void runOnOperation() override final;
};

}  // namespace

void AllPredecessors::print(llvm::raw_ostream& os) const {
  os << "AllPreds:" << getPoint() << ": ";
  if (blocks) {
    os << "[";
    for (auto b : *blocks) {
      os << b << ",";
    }
    os << "]";
  } else {
    os << "OVERDETERMINED";
  }
}

mlir::ChangeResult AllPredecessors::join(
    std::optional<DenseSet<Block*>> const& rhs) {
  if (!blocks) {
    return ChangeResult::NoChange;
  }
  if (!rhs) {
    blocks = rhs;
    return ChangeResult::Change;
  }
  ChangeResult cr = ChangeResult::NoChange;
  for (auto b : *rhs) {
    if (blocks->insert(b).second) {
      cr = ChangeResult::Change;
    }
  }
  return cr;
}

mlir::LogicalResult AllPredecessorsAnalysis::initialize(Operation* top) {
  for (auto& r : top->getRegions()) {
    for (auto& b : r.getBlocks()) {
      if (failed(visit(&b))) {
        return failure();
      }
      for (auto& o : b.getOperations()) {
        if (failed(initialize(&o))) {
          return failure();
        }
      }
    }
  }
  return success();
}

mlir::LogicalResult AllPredecessorsAnalysis::visit(ProgramPoint pp) {
  auto b = llvm::dyn_cast<Block*>(pp);
  if (!b) {
    return emitError(pp.getLoc()) << "Program point is not a block";
  }
  auto* all_preds = getOrCreate<AllPredecessors>(pp);
  ChangeResult cr = ChangeResult::NoChange;
  auto* predecessors = getOrCreate<dataflow::PredecessorState>(pp);
  if (predecessors->allPredecessorsKnown()) {
    for (auto op : predecessors->getKnownPredecessors()) {
      cr !=
          all_preds->join(getOrCreateFor<AllPredecessors>(pp, op->getBlock()));
    }
  } else {
    cr = all_preds->join(std::nullopt);
  }
  propagateIfChanged(all_preds, cr);
  return success();
}

hugr_mlir::VerifyLinearityAnalysis::VerifyLinearityAnalysis(
    mlir::Operation* top_, mlir::AnalysisManager&)
    : top(top_) {}

hugr_mlir::VerifyLinearityAnalysis::BlockSet_t const&
hugr_mlir::VerifyLinearityAnalysis::getPreds(mlir::Operation* op) {
  return solver->getOrCreateState<AllPredecessors>(op->getBlock())->get();
}

mlir::LogicalResult hugr_mlir::VerifyLinearityAnalysis::initialise() {
  if (solver) {
    return mlir::success();
  }

  solver = std::make_unique<mlir::DataFlowSolver>();
  solver->load<mlir::dataflow::SparseConstantPropagation>();
  solver->load<AllPredecessorsAnalysis>();
  auto r = solver->initializeAndRun(top);
  if (mlir::failed(r)) {
    solver = nullptr;
  }
  return r;
}

void HugrVerifyLinearityPass::runOnOperation() {
  auto& a = getAnalysis<hugr_mlir::VerifyLinearityAnalysis>();
  if (mlir::failed(a.initialise())) {
    return signalPassFailure();
  }

  mlir::SmallVector<mlir::Value> linear_sources;

  auto try_value = [&linear_sources](mlir::Value v) -> bool {
    if (auto t = llvm::dyn_cast<hugr_mlir::HugrTypeInterface>(v.getType())) {
      if (t.getConstraint() == hugr_mlir::TypeConstraint::Linear) {
        linear_sources.push_back(v);
        return true;
      }
    }
    return false;
  };

  getOperation()->walk([&](mlir::Operation* op) {
    for (auto const& o : op->getResults()) {
      try_value(o);
    }
    for (auto& r : op->getRegions()) {
      for (auto& b : r.getBlocks()) {
        for (auto a : b.getArguments()) {
          try_value(a);
        }
      }
    }
  });

  for (auto v : linear_sources) {
    DenseSet<Block*> all_preds;
    for (auto u : v.getUsers()) {
      if (auto bs = a.getPreds(u)) {
        for (auto b : *bs) {
          if (!all_preds.insert(b).second) {
            emitError(u->getLoc()) << "Failed to verify linearity";
            return signalPassFailure();
          }
        }
      } else {
        emitError(u->getLoc()) << "Failed to verify linearity";
      }
    }
  }
}
