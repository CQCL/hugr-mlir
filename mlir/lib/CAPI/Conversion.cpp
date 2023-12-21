#include "hugr-mlir-c/Conversion.h"

#include "hugr-mlir/Conversion/Passes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"

using namespace hugr_mlir;

#ifdef __cplusplus
extern "C" {
#endif

#include "hugr-mlir/Conversion/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
