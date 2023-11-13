#include "hugr-mlir/Analysis/VerifyLinearityPass.h"

#include "hugr-mlir/Analysis/FreeAllocAnalysis.h"

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
    auto& a = getAnalysis<hugr_mlir::FreeAllocAnalysis>();
    if(mlir::failed(a.initialise())) {
        return signalPassFailure();
    }
    llvm::outs() << "HugrVerifyLinearityPass\n";
}
