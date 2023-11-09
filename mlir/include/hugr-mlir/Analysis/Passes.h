#ifndef HUGR_MLIR_ANALYSIS_PASSES_H
#define HUGR_MLIR_ANALYSIS_PASSES_H

#include "mlir/Pass/Pass.h"


namespace hugr_mlir {
#define GEN_PASS_DECL_HUGRVERIFYLINEARITYPASS
#include "hugr-mlir/Analysis/Passes.h.inc"
}


namespace hugr_mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "hugr-mlir/Analysis/Passes.h.inc"

} // namespace mlir


#endif
