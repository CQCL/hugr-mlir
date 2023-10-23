// RUN: test-hugr-mlir-capi 2>&1 | FileCheck %s
#include <stdio.h>
#include <string.h>

#include "hugr-mlir-c/Dialects.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Debug.h"

int main() {
  mlirEnableGlobalDebug(true);
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  MlirDialectHandle hugr = mlirGetDialectHandle__hugr__();
  mlirDialectHandleInsertDialect(hugr, registry);

  MlirContext ctx = mlirContextCreateWithRegistry(registry, false);
  mlirDialectHandleLoadDialect(hugr, ctx);
  MlirLocation unknown = mlirLocationUnknownGet(ctx);

  MlirNamedAttribute attrs[2];

  const char* linear = "Linear";
  MlirAttribute test_attr = mlirHugrTypeConstraintAttrGet(
      ctx, mlirStringRefCreate(linear, strlen(linear)));
  const char* test_attr_str = "test_attr";
  MlirStringRef test_attr_strref =
      mlirStringRefCreate(test_attr_str, strlen(test_attr_str));
  MlirIdentifier test_attr_id = mlirIdentifierGet(ctx, test_attr_strref);
  attrs[0] = mlirNamedAttributeGet(test_attr_id, test_attr);

  MlirType test_type = mlirHugrSumTypeGet(ctx, 0, NULL);
  MlirAttribute test_type_attr = mlirTypeAttrGet(test_type);
  const char* test_type_attr_str = "test_type_attr";
  MlirStringRef test_type_attr_strref =
      mlirStringRefCreate(test_type_attr_str, strlen(test_type_attr_str));
  MlirIdentifier test_type_attr_id =
      mlirIdentifierGet(ctx, test_type_attr_strref);
  attrs[1] = mlirNamedAttributeGet(test_type_attr_id, test_type_attr);

  const char* hugr_module_str = "hugr.module";
  MlirStringRef hugr_module_strref =
      mlirStringRefCreate(hugr_module_str, strlen(hugr_module_str));

  MlirOperationState op_state =
      mlirOperationStateGet(hugr_module_strref, unknown);
  MlirRegion region = mlirRegionCreate();
  MlirBlock block = mlirBlockCreate(0, NULL, NULL);
  mlirRegionAppendOwnedBlock(region, block);
  mlirOperationStateAddOwnedRegions(&op_state, 1, &region);
  mlirOperationStateAddAttributes(&op_state, 2, attrs);

  MlirOperation op = mlirOperationCreate(&op_state);

  mlirOperationDump(op);
  // CHECK: hugr.module attributes {
  // CHECK: test_attr = #hugr<constraint Linear>,
  // CHECK: test_type_attr = !hugr.sum<>} {

  mlirOperationDestroy(op);
  mlirContextDestroy(ctx);
  return 0;
}
