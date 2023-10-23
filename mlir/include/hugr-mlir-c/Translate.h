#ifndef HUGR_MLIR_C_TRANSLATE_H
#define HUGR_MLIR_C_TRANSLATE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef MlirOperation (*mlirHugrTranslateStringRefToMLIRFunction)(
    MlirStringRef, MlirLocation);
typedef void (*mlirHugrDialectRegistrationFunction)(MlirDialectRegistry);

MLIR_CAPI_EXPORTED
void mlirHugrRegisterTranslationToMLIR(
    MlirStringRef name, MlirStringRef description,
    mlirHugrTranslateStringRefToMLIRFunction,
    mlirHugrDialectRegistrationFunction);

MLIR_CAPI_EXPORTED
int mlirHugrTranslateMain(int argc, char const* const* argv);

#ifdef __cplusplus
}
#endif

#endif  // HUGR_MLIR_C_TRANSLATE_H
