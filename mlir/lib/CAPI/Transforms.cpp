#include "hugr-mlir-c/Transforms.h"

#include "hugr-mlir/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"

using namespace hugr_mlir;

#ifdef __cplusplus
extern "C" {
#endif

#include "hugr-mlir/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
