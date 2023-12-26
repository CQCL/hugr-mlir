#ifndef HUGR_MLIR_TRANSFORMS_PASSES_TD
#define HUGR_MLIR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

//===----------------------------------------------------------------------===//
// ToLLVM
//===----------------------------------------------------------------------===//

def LowerHugrPass : Pass<"lower-hugr"> {
  let summary = "TODO";
  let description = [{
  TODO
  }];

  let options = RewritePassUtils.options # [
      Option<"hugrVerify", "hugr-verify", "bool", "false", "Do additional, potentially expensive, verification">
  ];

  let dependentDialects = [
    "hugr_mlir::HugrDialect",
    "mlir::func::FuncDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
    "mlir::index::IndexDialect",
  ];
}

#endif
