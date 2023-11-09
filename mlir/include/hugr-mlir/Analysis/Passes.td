
#ifndef HUGR_MLIR_ANALYSIS_PASSES
#define HUGR_MLIR_ANALYSIS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ToLLVM
//===----------------------------------------------------------------------===//

def HugrVerifyLinearityPass : Pass<"hugr-verify-linearity"> {
  let summary = "TODO";
  let description = [{
  TODO
  }];

  let options = [
  ];
}

#endif
