//===- HugrTypes.td - Hugr dialect types -------*- tablegen -*-===//

#ifndef HUGR_MLIR_IR_HUGR_TYPES_TD
#define HUGR_MLIR_IR_HUGR_TYPES_TD

include "hugr-mlir/IR/HugrDialect.td"
include "hugr-mlir/IR/HugrAttrs.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def Hugr_ClosureType : Hugr_Type<"Closure","closure",[MemRefElementTypeInterface]> {
    let summary = "A Hugr Closure Type";
    let description = [{ TODO }];

    let assemblyFormat = "";
}

def Hugr_FunctionType : Hugr_Type<"Function", "function"> {
    let summary = "A Hugr Function Type";
    let description = [{
        Corresponds to `hugr::signature::FunctionType` in the hugr crate.

        We do not verify that all input and output types implement
        `HugrTypeInterface` because external passes may modify them.
        Instead, another pass (unimplemented) should be used to either error or
        clean up the types.

        Implements `HugrTypeInterface`, although this is implemented outside of
        tablegen.
    }];
    let assemblyFormat = "$extensions``$function_type";

    let parameters = (ins
        ExtensionSetParameter<"extensions">:$extensions,
        TypeParameter<"::mlir::FunctionType","function_type">:$function_type
    );

    let builders = [
        TypeBuilderWithInferredContext<(ins "::hugr_mlir::ExtensionSetAttr":$extensions, "::mlir::FunctionType":$function_type), [{
          return $_get(function_type.getContext(), extensions, function_type);
        }]>
    ];

    let extraClassDeclaration = [{
        FunctionType clone(mlir::TypeRange,mlir::TypeRange);
        mlir::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }
        mlir::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
    }];
   let genVerifyDecl = 1;
}

def Hugr_OpaqueType : Hugr_Type<"Opaque", "opaque"
   > {
    let summary = "A Hugr Opaque type";
    let description = [{
        Corresponds to `hugr::types::custom::CustomType` in the hugr crate.

        Implements `HugrTypeInterface`, although this is implemented outside of
        tablegen.

        TODO for now we take type_args as Attributes. We'll need to iterate on
        this as it comes up
    }];
    let assemblyFormat = "$name`[`$extension`]`(`[`$constraint^`]`)?(`<`$type_args^`>`)?";
    let parameters = (ins
        StringRefParameter<"type name">:$name,
        TypeParameter<"hugr_mlir::ExtensionAttr","extension">:$extension,
        OptionalArrayRefParameter<"::mlir::Attribute", "type args">:$type_args,
        DefaultValuedParameter<"TypeConstraint","TypeConstraint::Equatable">:$constraint
    );
    // TODO type args can be usize, type, extensionset, or SumType, TupleType, FunctionType, OpaqueType
    // Q: Can they be extended types?
   // let genVerifyDecl = 1;
   let builders = [
       TypeBuilderWithInferredContext<(ins "llvm::StringRef":$name, "hugr_mlir::ExtensionAttr":$extension, "llvm::ArrayRef<mlir::Attribute>":$args, CArg<"hugr_mlir::TypeConstraint","hugr_mlir::TypeConstraint::Equatable">:$constraint), [{
           return $_get(extension.getContext(), name, extension, args, constraint);
       }]>
   ];
}

def Hugr_SumType : Hugr_Type<"Sum", "sum",[]> {
    let summary = "A Hugr Sum type";
    let description = [{
        Corresponds to a `hugr::types::SumType` in the hugr crate.

        Implements `HugrTypeInterface`, although this is implemented outside of
        tablegen.

        We do not verify that component types are HugrTypeInterface. See
        description of HugrTypeInterface.
    }];
    let parameters = (ins
        OptionalArrayRefParameter<"::mlir::Type", "component types">:$types
    );
    let assemblyFormat = "`<`$types`>`";
    let extraClassDeclaration = [{
        unsigned numAlts() { return getTypes().size(); }
        mlir::Type getAltType(unsigned i) { return getTypes()[i]; }
    }];
}

def PredicateType : Type<
    CPred<"llvm::isa<SumType>($_self) && llvm::all_of(llvm::cast<SumType>($_self).getTypes(), [](auto x) { return llvm::isa<mlir::TupleType>(x); })">,
    "A hugr predicate type",
    "::hugr_mlir::SumType">;


def Hugr_ExtendedType : Hugr_Type<"Extended", "ext"> {
    let summary = "A Hugr type annotated with extensions";
    let description = [{
        An arbitrary type annoted with an extension set and a type constraint.
        The inner type MUST NOT be another ExtendedType.

        One can build an ExtendedType from any `HugrTypeInterface`.

        Implements `HugrTypeInterface`, although this is implemented outside of
        tablegen.

        TODO Except for special cases (i.e. SumType, TupleType, OpaqueType,
        TypeAliasType), the inner type maps to an "mlir.ext_type" extension type
        in hugr.
    }];
    let parameters = (ins
        ExtensionSetParameter<"extensions">:$extensions,
        TypeParameter<"::mlir::Type","inner type">:$inner_type,
        DefaultValuedParameter<"TypeConstraint","TypeConstraint::Equatable">:$constraint
    );
    let assemblyFormat = "`<`$extensions``$inner_type``(`,`$constraint^)?`>`";
    let builders = [
        TypeBuilderWithInferredContext<(ins "HugrTypeInterface":$t), [{
          if(auto et = llvm::dyn_cast<ExtendedType>(t)) { return et; }
          return $_get(t.getContext(), ExtensionSetAttr::get(t.getContext()), t, t.getConstraint());
        }]>
    ];

    let extraClassDeclaration = [{
        ExtendedType removeExtensions(ExtensionSetAttr);
    }];
}

def Hugr_AliasRefType : Hugr_Type<"AliasRef","ref"> {
    let summary = "A symbol reference to TypeAliasOp";
    let description = [{
        References a TypeAliasOp.

        Implements `HugrTypeInterface`, although this is implemented outside of
        tablegen.

        TODO Hugr Ops should implement SymbolOpUserInterface and verify any
        AliasRefTypes they contain .
    }];
    let assemblyFormat = "`<`$extensions``$ref(`,`$constraint^)?`>`";
    let parameters = (ins
        ExtensionSetParameter<"Extension">:$extensions,
        TypeParameter<"mlir::SymbolRefAttr","ref">:$ref,
        DefaultValuedParameter<"TypeConstraint","TypeConstraint::Equatable">:$constraint
    );

    let builders = [
        TypeBuilderWithInferredContext<(ins "hugr_mlir::ExtensionSetAttr":$extensions, "mlir::SymbolRefAttr":$ref, CArg<"hugr_mlir::TypeConstraint","hugr_mlir::TypeConstraint::Equatable">:$constraint), [{
          return $_get(extensions.getContext(), extensions, ref, constraint);
        }]>
    ];

}

#endif
