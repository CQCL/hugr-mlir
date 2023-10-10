#ifndef HUGR_MLIR_CAPI_DIALECTS_H
#define HUGR_MLIR_CAPI_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Hugr, hugr);

// TODO Add *{Attr,Type}Get functions for each custom type and attribute
MLIR_CAPI_EXPORTED bool
mlirTypeIsAHugrSumType(MlirType);

MLIR_CAPI_EXPORTED MlirType
mlirHugrSumTypeGet(MlirContext, int32_t n, MlirType const* components);

MLIR_CAPI_EXPORTED bool
mlirTypeIsAHugrFunctionType(MlirType);

MLIR_CAPI_EXPORTED MlirType
mlirHugrFunctionTypeGet(MlirAttribute extensions, MlirType function_type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsAHugrAliasRefType(MlirType);

MLIR_CAPI_EXPORTED MlirType
mlirHugrAliasRefTypeGet(MlirAttribute extensions, MlirAttribute sym_ref, MlirAttribute type_constraint);

MLIR_CAPI_EXPORTED bool
mlirTypeIsAHugrOpaqueType(MlirType);

MLIR_CAPI_EXPORTED MlirType
mlirHugrOpaqueTypeGet(MlirStringRef name, MlirAttribute extension, MlirAttribute type_constraint, intptr_t n_args, MlirAttribute const* args);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAHugrExtensionAttr(MlirAttribute);

MLIR_CAPI_EXPORTED MlirAttribute
mlirHugrExtensionAttrGet(MlirContext, MlirStringRef);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAHugrExtensionSetAttr(MlirAttribute);

MLIR_CAPI_EXPORTED MlirAttribute
mlirHugrExtensionSetAttrGet(MlirContext, int32_t n_extensions, MlirAttribute const* extensions);

MLIR_CAPI_EXPORTED bool
mlirAttrIsAHugrTypeConstraintAttr(MlirAttribute);

MLIR_CAPI_EXPORTED MlirAttribute
mlirHugrTypeConstraintAttrGet(MlirContext, MlirStringRef kind);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsHugrStaticEdgeAttr(MlirAttribute);

MLIR_CAPI_EXPORTED MlirAttribute
mlirHugrStaticEdgeAttrGet(MlirType type, MlirAttribute sym);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsHugrSumAttr(MlirAttribute);

MLIR_CAPI_EXPORTED MlirAttribute
mlirHugrSumAttrGet(MlirType type, uint32_t tag, MlirAttribute value);


#ifdef __cplusplus
}
#endif

#endif
