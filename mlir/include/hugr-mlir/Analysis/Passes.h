#ifndef HUGR_MLIR_ANALYSIS_PASSES_H
#define HUGR_MLIR_ANALYSIS_PASSES_H

#include "hugr-mlir/Analysis/VerifyLinearityPass.h"

namespace hugr_mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "hugr-mlir/Analysis/Passes.h.inc"

} // namespace mlir


#endif
