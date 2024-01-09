#ifndef HUGR_MLIR_CONVERSION_PASSES_TD
#define HUGR_MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ConvertHugrModulePass : Pass<"convert-hugr-modules", "mlir::ModuleOp"> {
  let summary = "Lowers hugr.modules into their containing builtin.modules";
  let description = [{
  TODO
  }];
  let options = RewritePassUtils.options # [
  ];
  let dependentDialects = [
    "::hugr_mlir::HugrDialect",
  ];
}

def PreConvertHugrFuncPass : Pass<"pre-convert-hugr-func","hugr_mlir::ModuleOp"> {
  let summary = "Ensures all functions have no non-local edges";
  let description = [{
      Lower the hugr module as far as possible while maintaining the ability to convert back to hugr.

      Eliminates switch.

      Ensures all hugr.func ops have no non-local edges by adding explicit captures.

      Ensures all calls to non-top-level functions are through function values, not symbols.
  }];

  let options = RewritePassUtils.options # [
  ];

  let dependentDialects = [
    "::hugr_mlir::HugrDialect",
    "::mlir::func::FuncDialect",
    "::mlir::index::IndexDialect",
    "::mlir::cf::ControlFlowDialect",
  ];
}

def ConvertHugrFuncPass : Pass<"convert-hugr-func","hugr_mlir::ModuleOp"> {
  let summary = "Convert all hugr.func to func.func";
  let description = [{
      Moves all funcs to the top level, and convert to func.func

      Eliminates !hugr.function type, replaced by tuple<function,!hugr.closure>

      Eliminates hugr.alloc_function, hugr.func, hugr.call
  }];

  let options = RewritePassUtils.options # [
  ];

  let dependentDialects = [
    "::hugr_mlir::HugrDialect",
    "::mlir::func::FuncDialect",
    "::mlir::index::IndexDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::ub::UBDialect",
  ];
}

def ConvertHugrPass : Pass<"convert-hugr"> {
  let summary = "TODO";
  let description = [{
  TODO
  }];

  let options = RewritePassUtils.options # [
      Option<"hugrVerify", "hugr-verify", "bool", "true", "Do additional, potentially expensive, verification">
  ];

  let dependentDialects = [
    "::hugr_mlir::HugrDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::ub::UBDialect",
    "::mlir::index::IndexDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

#endif
