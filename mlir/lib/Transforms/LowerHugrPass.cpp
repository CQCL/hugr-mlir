#include "hugr-mlir/Transforms/LowerHugrPass.h"


namespace hugr_mlir {
#define GEN_PASS_DEF_LOWERHUGRPASS
#include "hugr-mlir/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

struct LowerHugrPass : hugr_mlir::impl::LowerHugrPassBase<LowerHugrPass> {
    void runOnOperation() override;
};
}


void LowerHugrPass::runOnOperation() {

}
