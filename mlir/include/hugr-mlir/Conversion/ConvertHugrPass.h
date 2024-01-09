#ifndef HUGR_MLIR_CONVERSION_CONVERT_HUGR_H
#define HUGR_MLIR_CONVERSION_CONVERT_HUGR_H

#include "mlir/Pass/Pass.h"

namespace hugr_mlir {
#define GEN_PASS_DECL_CONVERTHUGRPASS
#define GEN_PASS_DECL_CONVERTHUGRMODULEPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir

#endif  // HUGR_MLIR_CONVERSION_CONVERT_HUGR_TO_H
