#ifndef HUGR_MLIR_TRANSFORMS_PASSES_H
#define HUGR_MLIR_TRANSFORMS_PASSES_H

namespace hugr_mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "hugr-mlir/Transforms/Passes.h.inc"

}  // namespace hugr_mlir

#endif
