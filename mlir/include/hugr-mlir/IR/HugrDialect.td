//===- HugrDialect.td - Hugr dialect -----------*- tablegen -*-===//

#ifndef HUGR_MLIR_IR_HUGR_DIALECT_TD
#define HUGR_MLIR_IR_HUGR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
//===----------------------------------------------------------------------===//
// Hugr dialect definition.
//===----------------------------------------------------------------------===//

def Hugr_Dialect : Dialect {
    let name = "hugr";
    let summary = "A dialect representing hugr graphs";
    let description = [{
        A dialect intended to facilitate importing and exporting [hugr](https://github.com/CQCL-DEV/hugr/blob/main/specification/hugr.md)
        through mlir.

        We model the full `hugr` type system in mlir types. We intend to
        delegate to the `hugr` crate whenever possible through the companion
        rust crate.

        Linear Types
        ============
        We model `hugr` type constraints, so we know which types are linear.
        We will likely try both "reference" style linear types a la qir, as well
        as "ssa value" style linear types, like hugr itself.

        Dataflow Sibling Graphs
        =======================
        Several hugr ops contain dataflow sibling graphs. The CFG op contains a
        control flow sibling graph. Both dataflow sibling graphs and control
        flow sibling graphs can contain scoped definitions.

        We model operations containing dataflow sibling graphs as `SingleBlock`
        ops and (TODO) verify that they are they are terminated with a
        `hugr.output` op. We permit these ops to contain `hugr.func`,
        `hugr.const`, and `hugr.type_alias` ops. Note that this does not
        directly correspond to the representation in hugr, because the scoped
        definitions may capture values defined inside the `SingleBlock` region.
        We do not permit(TODO) `hugr.cfg` ops to contain scoped definitions.

        Extension Ops
        =============
        the `hugr.ext_op` op models an arbitrary hugr extension node. (TODO) We
        define an `mlir` extension in the companion rust crate, and we so we are
        able to convert an arbitary no-successor, no-region, mlir op into a hugr
        node; this does require serialising custom mlir attributes and types to
        strings.
    }];
    let cppNamespace = "::hugr_mlir";

    let extraClassDeclaration = [{
        void registerTypes();
        void registerOps();
        void registerAttrs();
        void registerTypeInterfaces();
    }];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

    let dependentDialects = [
    ];
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Hugr_Op<string mnemonic, list<Trait> traits = []> :
        Op<Hugr_Dialect, mnemonic, traits>;

class Hugr_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Hugr_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class Hugr_Type<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Hugr_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

class Hugr_EnumAttr<EnumAttrInfo enum, string mnemonic, list<Trait> traits = []>
    : EnumAttr<Hugr_Dialect, enum, mnemonic, traits>;

class Hugr_TypeInterface<string name> : TypeInterface<name> {
  let cppNamespace = Hugr_Dialect.cppNamespace;
}

#endif
