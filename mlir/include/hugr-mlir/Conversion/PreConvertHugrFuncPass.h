#ifndef HUGR_MLIR_CONVERSION_PRECONVERTHUGRFUNCPASS_H
#define HUGR_MLIR_CONVERSION_PRECONVERTHUGRFUNCPASS_H

#include "mlir/Pass/Pass.h"

namespace hugr_mlir {
#define GEN_PASS_DECL_PRECONVERTHUGRFUNCPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir


#endif // HUGR_MLIR_CONVERSION_PRECONVERTHUGRFUNCPASS_H
