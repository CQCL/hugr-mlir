#include "hugr-mlir-c/Opt.h"
#include "hugr-mlir/Opt/HugrOpt.h"

void mlirHugrRegisterOptPipelines() {
    hugr_mlir::registerHugrOptPipelines();
}
