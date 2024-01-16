#ifndef HUGR_MLIR_CONVERSION_CONVERT_HUGR_EXT_ARITH_PASS_H
#define HUGR_MLIR_CONVERSION_CONVERT_HUGR_EXT_ARITH_PASS_H
#include "mlir/Pass/Pass.h"

namespace hugr_mlir {
#define GEN_PASS_DECL_CONVERTHUGREXTARITHPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir

#endif
