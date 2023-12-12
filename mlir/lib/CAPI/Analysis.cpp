#include "hugr-mlir-c/Analysis.h"

#include "hugr-mlir/Analysis/Passes.h"
#include "mlir/CAPI/Pass.h"

using namespace hugr_mlir;

#ifdef __cplusplus
extern "C" {
#endif

#include "hugr-mlir/Analysis/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
