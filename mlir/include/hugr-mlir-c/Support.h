#ifndef HUGR_MLIR_C_SUPPORT_H
#define HUGR_MLIR_C_SUPPORT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED unsigned mlirOperationHash(MlirOperation);
MLIR_CAPI_EXPORTED unsigned mlirTypeHash(MlirType);
MLIR_CAPI_EXPORTED unsigned mlirAttributeHash(MlirAttribute);
MLIR_CAPI_EXPORTED unsigned mlirBlockHash(MlirBlock);
MLIR_CAPI_EXPORTED unsigned mlirRegionHash(MlirRegion);
MLIR_CAPI_EXPORTED unsigned mlirValueHash(MlirValue);

#ifdef __cplusplus
}
#endif

#endif
