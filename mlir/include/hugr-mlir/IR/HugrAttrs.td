//===- HugrAttrs.td - Hugr dialect attributes -------*- tablegen -*-===//

#ifndef HUGR_MLIR_IR_HUGR_ATTRS_TD
#define HUGR_MLIR_IR_HUGR_ATTRS_TD

include "hugr-mlir/IR/HugrDialect.td"
include "hugr-mlir/IR/HugrEnums.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Hugr_ExtensionAttr : Hugr_Attr<"Extension", "ext"> {
    let summary = "a hugr extension atom";
    let description = [{
        Many entities in the `hugr` dialect carry an extension, or a set of
        extensions.
    }];

    let parameters = (ins
        StringRefParameter<"extension name">:$name
    );
    let assemblyFormat = "``$name";
    let extraClassDeclaration = [{
      int compare(ExtensionAttr rhs) { return getName().compare(rhs.getName()); }
    }];
}

class ExtensionListParameter<string desc> : OptionalArrayRefParameter<"hugr_mlir::ExtensionAttr", desc> {
    let parser = "hugr_mlir::ExtensionSetAttr::parseExtensionList($_parser)";
    let printer = [{
        bool first = true;
        for(auto ext: $_self) {
            if(first) { first = false; } else { $_printer << ","; }
            $_printer.printStrippedAttrOrType(ext);
        }
    }];
}

def Hugr_ExtensionSetAttr : Hugr_Attr<"ExtensionSet", "exts"> {
  let summary = "A set of hugr extensions";
  let description = [{
      A set of hugr extensions.

      This attribute is truly a set, and so the internal extensions are stored
      sorted, and with duplicates removed.
  }];
  let assemblyFormat = "`[```$extensions`]`";
  let parameters = (ins
      ExtensionListParameter<"extension set">:$extensions
  );
  let skipDefaultBuilders = 1;
  let builders = [
      AttrBuilder<(ins CArg<"::mlir::ArrayRef<ExtensionAttr>","{}">:$extensions), [{
        mlir::SmallVector<ExtensionAttr> v;
        mlir::DenseSet<ExtensionAttr> s;
        for(auto x: extensions) {
          if(s.insert(x).second) {
            v.push_back(x);
          }
        }
        llvm::sort(v,[](auto x, auto y) { return x.compare(y); });
        return Base::$_get($_ctxt, v);
      }]>
  ];
  let extraClassDeclaration = [{
      static mlir::FailureOr<mlir::SmallVector<ExtensionAttr>> parseExtensionList(mlir::AsmParser&);
      unsigned size() { return getExtensions().size(); }
      ExtensionSetAttr remove(ExtensionSetAttr rhs);
  }];
}

class ExtensionSetParameter<string desc>
      : AttrOrTypeParameter<"hugr_mlir::ExtensionSetAttr", desc> {
}

def Hugr_StaticEdgeAttr: Hugr_Attr<"StaticEdge", "static_edge",
    [TypedAttrInterface]> {
    let summary = "A reference to a `hugr.func`, or `hugr.const`";
    let description = [{
        A typed `SymbolRefAttr`.
    }];
    let assemblyFormat = "`<`$ref`>`";
    let parameters = (ins
        AttributeSelfTypeParameter<"type">:$type,
        AttrParameter<"mlir::SymbolRefAttr", "ref">:$ref
    );
    let genVerifyDecl = 1;
    let builders = [
        AttrBuilderWithInferredContext<(ins "mlir::Type":$type, "mlir::SymbolRefAttr":$ref), [{
          return $_get(type.getContext(), type, ref);
        }]>
    ];
}

def Hugr_UnitAttr : Hugr_Attr<"Unit", "unit",
    [TypedAttrInterface]> {
    let summary = "The single inhabitant of tuple<>";
    let description = [{
        A representation of a compile-time-constant Value of type tuple<>

        TODO we should use a more general Tuple attribute type. Likely
        `ArrayAttr` will be fine.
    }];
    let assemblyFormat = "";
    let extraClassDeclaration = [{
        mlir::TupleType getType();
    }];
}

def Hugr_SumAttr : Hugr_Attr<"Sum", "sum",
    [TypedAttrInterface]> {
    let summary = "The single inhabitant of tuple<>";
    let description = [{
        A representation of a compile-time-constant Value of type tuple<>

        TODO we should use a more general Tuple attribute type. Likely
        `ArrayAttr` will be fine.
    }];
    let assemblyFormat = "`<`$tag`,`$value`>`";
    let parameters = (ins
        AttributeSelfTypeParameter<"type">:$type,
        AttrParameter<"uint32_t", "tag">:$tag,
        AttrParameter<"mlir::Attribute","value">:$value
    );
    let builders = [
        AttrBuilderWithInferredContext<(ins "hugr_mlir::SumType":$type, "uint32_t":$tag, "mlir::Attribute":$value), [{
          return $_get(type.getContext(), type, tag, value);
        }]>

    ];
    let extraClassDeclaration = [{
        hugr_mlir::SumType getSumType();
    }];
}

def Hugr_TypeConstraintAttr : Hugr_EnumAttr<Hugr_TypeConstraintEnum, "constraint", []> {
    let summary = "The set of constraints on a type";
    let description = [{
        For now we use an enum to represent a set of type constraints.
    }];
    let assemblyFormat = "$value";
    let extraClassDeclaration = [{
        TypeConstraintAttr intersection(TypeConstraintAttr);
    }];
}

#endif
