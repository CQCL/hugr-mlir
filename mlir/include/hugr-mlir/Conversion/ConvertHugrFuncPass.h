#ifndef HUGR_MLIR_CONVERSION_CONVERTHUGRFUNCPASS_H
#define HUGR_MLIR_CONVERSION_CONVERTHUGRFUNCPASS_H

#include "mlir/Pass/Pass.h"

namespace hugr_mlir {
#define GEN_PASS_DECL_PRECONVERTHUGRFUNCPASS
#define GEN_PASS_DECL_CONVERTHUGRFUNCPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir

#endif  // HUGR_MLIR_CONVERSION_CONVERTHUGRFUNCPASS_H
