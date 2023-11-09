#include "hugr-mlir/Analysis/Passes.h"

namespace hugr_mlir {
#define GEN_PASS_DEF_HUGRVERIFYLINEARITYPASS
#include "hugr-mlir/Analysis/Passes.h.inc"
} // namespace mlir

namespace {

struct HugrVerifyLinearityPass : hugr_mlir::impl::HugrVerifyLinearityPassBase<HugrVerifyLinearityPass> {
    void runOnOperation() override final;
};

}

void HugrVerifyLinearityPass::runOnOperation() {
    llvm::outs() << "HugrVerifyLinearityPass\n";
}
