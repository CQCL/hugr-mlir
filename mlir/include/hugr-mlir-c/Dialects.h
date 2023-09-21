#ifndef HUGR_MLIR_CAPI_DIALECTS_H
#define HUGR_MLIR_CAPI_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Hugr, hugr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirHugrTypeConstraintAttrGet(MlirContext, const char* kind);
MLIR_CAPI_EXPORTED MlirType mlirHugrTestTypeGet(MlirContext);

#ifdef __cplusplus
}
#endif

#endif
