#ifndef HUGR_MLIR_CONVERSION_PASSES_TD
#define HUGR_MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ConvertHugrPass : Pass<"convert-hugr"> {
  let summary = "TODO";
  let description = [{
  TODO
  }];

  let options = RewritePassUtils.options # [
  ];

  let dependentDialects = [
    "::hugr_mlir::HugrDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::ub::UBDialect",
    "::mlir::index::IndexDialect",
  ];
}

#endif
