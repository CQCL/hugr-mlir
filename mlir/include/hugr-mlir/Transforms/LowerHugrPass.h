#ifndef HUGR_MLIR_LOWER_HUGR_PASS_H
#define HUGR_MLIR_LOWER_HUGR_PASS_H

#include "mlir/Pass/Pass.h"

namespace hugr_mlir {
#define GEN_PASS_DECL_LOWERHUGRPASS
#include "hugr-mlir/Analysis/Passes.h.inc"
}

#endif // HUGR_MLIR_LOWER_HUGR_PASS_H
