#ifndef HUGR_MLIR_CONVERSION_FINALIZEHUGRTOLLVMPASS_H
#define HUGR_MLIR_CONVERSION_FINALIZEHUGRTOLLVMPASS_H

#include "mlir/Pass/Pass.h"

namespace hugr_mlir {
#define GEN_PASS_DECL_FINALIZEHUGRTOLLVMPASS
#include "hugr-mlir/Conversion/Passes.h.inc"
}  // namespace hugr_mlir


#endif // HUGR_MLIR_CONVERSION_FINALIZEHUGRTOLLVMPASS_H
