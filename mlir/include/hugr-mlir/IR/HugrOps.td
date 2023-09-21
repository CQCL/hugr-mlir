//===- HugrOps.td - Hugr dialect ops -----------*- tablegen -*-===//

#ifndef HUGR_MLIR_IR_HUGR_OPS_TD
#define HUGR_MLIR_IR_HUGR_OPS_TD

include "hugr-mlir/IR/HugrDialect.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

def Hugr_ModuleOp : Hugr_Op<"module",
  [ SymbolTable, Symbol
  , NoTerminator, SingleBlock, NoRegionArguments
  , OpAsmOpInterface
  ]> {
    let summary = "Hugr module op";
    let description = "TODO";
    let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $body";

    let arguments = (ins
        OptionalAttr<SymbolNameAttr>:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility
    );

    let regions = (region AnyRegion:$body);
    let extraClassDeclaration = [{
        static mlir::StringRef getDefaultDialect() { return "hugr"; }
        bool isOptionalSymbol() { return true; }
    }];
}

#endif
