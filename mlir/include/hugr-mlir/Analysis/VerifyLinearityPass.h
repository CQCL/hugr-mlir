#ifndef HUGR_MLIR_ANALYSIS_VERIFY_LINEARITY_PASS
#define HUGR_MLIR_ANALYSIS_VERIFY_LINEARITY_PASS

#include "mlir/Pass/Pass.h"

namespace hugr_mlir {
#define GEN_PASS_DECL_HUGRVERIFYLINEARITYPASS
#include "hugr-mlir/Analysis/Passes.h.inc"
}


#endif // HUGR_MLIR_ANALYSIS_VERIFY_LINEARITY_PASS
