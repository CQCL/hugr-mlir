#ifndef HUGR_MLIR_ANALYSIS_VERIFY_LINEARITY_PASS
#define HUGR_MLIR_ANALYSIS_VERIFY_LINEARITY_PASS

#include "mlir/Pass/Pass.h"

namespace hugr_mlir {
#define GEN_PASS_DECL_HUGRVERIFYLINEARITYPASS
#include "hugr-mlir/Analysis/Passes.h.inc"
}  // namespace hugr_mlir

namespace mlir {
class DataFlowSolver;
}

namespace hugr_mlir {

struct VerifyLinearityAnalysis {
  using BlockSet_t = std::optional<mlir::DenseSet<mlir::Block*>>;
  VerifyLinearityAnalysis(mlir::Operation*, mlir::AnalysisManager&);
  mlir::LogicalResult initialise();
  BlockSet_t const& getPreds(mlir::Operation*);

 private:
  std::unique_ptr<mlir::DataFlowSolver> solver;
  mlir::Operation* top;
};

}  // namespace hugr_mlir

#endif  // HUGR_MLIR_ANALYSIS_VERIFY_LINEARITY_PASS
