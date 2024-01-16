#ifndef HUGR_MLIR_CONVERSION_PASSES_H
#define HUGR_MLIR_CONVERSION_PASSES_H

#include "hugr-mlir/Conversion/FinalizeHugrToLLVMPass.h"
#include "hugr-mlir/Conversion/PreConvertHugrFuncPass.h"
#include "hugr-mlir/Conversion/ConvertHugrFuncPass.h"
#include "hugr-mlir/Conversion/ConvertHugrPass.h"
#include "hugr-mlir/Conversion/ConvertHugrExtArithPass.h"
#include "mlir/Pass/Pass.h"

namespace hugr_mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "hugr-mlir/Conversion/Passes.h.inc"

}  // namespace hugr_mlir

#endif
