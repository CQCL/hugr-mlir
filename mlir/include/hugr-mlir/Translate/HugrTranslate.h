#ifndef HUGR_MLIR_HUGR_TRANSLATE_TRANSLATE_H
#define HUGR_MLIR_HUGR_TRANSLATE_TRANSLATE_H

#include "mlir/Support/LogicalResult.h"

namespace hugr_mlir {

mlir::LogicalResult translateMain(int argc, char const* const* argv);

}

#endif
