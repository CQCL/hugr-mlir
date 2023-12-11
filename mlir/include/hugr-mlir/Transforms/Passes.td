
#ifndef HUGR_MLIR_TRANSFORMS_PASSES
#define HUGR_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ToLLVM
//===----------------------------------------------------------------------===//

def LowerHugrPass : Pass<"hugr-lower"> {
  let summary = "TODO";
  let description = [{
  TODO
  }];

  let options = [
  ];
}

#endif
