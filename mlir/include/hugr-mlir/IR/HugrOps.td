//===- HugrOps.td - Hugr dialect ops -----------*- tablegen -*-===//

#ifndef HUGR_MLIR_IR_HUGR_OPS_TD
#define HUGR_MLIR_IR_HUGR_OPS_TD

include "hugr-mlir/IR/HugrDialect.td"
include "hugr-mlir/IR/HugrTypes.td"
include "hugr-mlir/IR/HugrAttrs.td"
include "hugr-mlir/IR/HugrEnums.td"
include "hugr-mlir/IR/HugrTypeInterfaces.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DerivedAttributeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LinearityResource : Resource<"hugr_mlir::LinearityResource">;

def HugrTypeMemoryEffectsTrait : NativeOpTrait<"HugrTypeMemoryEffectsTrait">;
def HugrMemoryOp : TraitList<[MemoryEffectsOpInterface, HugrTypeMemoryEffectsTrait]>;

def HugrSymbolUserTrait : NativeOpTrait<"HugrSymbolUserTrait">;
def HugrSymbolUserOp : TraitList<[SymbolUserOpInterface, HugrSymbolUserTrait]>;

def DataflowGraphRegion : Region<
    CPred<"hugr_mlir::isDataflowGraphRegion($_self)">,
    "a single block region terminated by `hugr.output`">;

def ControlFlowGraphRegion : Region<
    CPred<"hugr_mlir::isControlFlowGraphRegion($_self)">,
    "a many-block region where all blocks without successors are terminated by matching `hugr.output`s">;

def Hugr_ModuleOp : Hugr_Op<"module",
  [ SymbolTable, Symbol, NoTerminator, SingleBlock, NoRegionArguments, IsolatedFromAbove
  , OpAsmOpInterface ]> {
    let summary = "Hugr module op";
    let description = [{
        Represents a `hugr::module::Module` node.
    }];
    let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $body";

    let arguments = (ins
        OptionalAttr<SymbolNameAttr>:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility
    );

    let regions = (region SizedRegion<1>:$body);


    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins CArg<"std::optional<llvm::StringRef>", "std::nullopt">:$name), [{
          $_state.addRegion()->emplaceBlock();
          if(name) {
             $_state.attributes.push_back($_builder.getNamedAttr("sym_name", $_builder.getStringAttr(*name)));
          }
        }]>

    ];

    let extraClassDeclaration = [{
        static mlir::StringRef getDefaultDialect() { return "hugr"; }
        bool isOptionalSymbol() { return true; }
    }];
}

def Hugr_FuncOp : Hugr_Op<"func",
  [ DeclareOpInterfaceMethods<CallableOpInterface>
  , FunctionOpInterface
  , Symbol, SingleBlock
  , OpAsmOpInterface
  , HugrSymbolUserOp
  ]> {
    let summary = "Hugr FuncDefn/FuncDecl op";
    let hasCustomAssemblyFormat = 1;
    let description = [{
        Represents a `hugr::module::FuncDecl` or `hugr::module::FuncDefn` node,
        depending on whether a body region is present.

        Note that this op is not `IsolatedFromAbove`, it may capture values from
        it's parent region.

        We do not fail to verify if `Linear` values are captured. Instead we
        will have a pass that verifies this.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        DefaultValuedStrAttr<StrAttr,"private">:$sym_visibility,
        TypeAttrOf<Hugr_FunctionType>:$function_type,
        OptionalAttr<ArrayAttr>:$arg_attrs,
        OptionalAttr<ArrayAttr>:$res_attrs
    );

    let regions = (region DataflowGraphRegion:$body);

    let extraClassDeclaration = [{
        bool isDeclaration() { return getBody().empty(); }
        static mlir::StringRef getDefaultDialect() { return "hugr"; }

        ExtensionSetAttr getExtensionSet() { return getFunctionType().getExtensions(); }
    }];
}

def Hugr_TypeAliasOp : Hugr_Op<"type_alias",
    [Symbol, HugrSymbolUserOp]> {
    let summary = "A Hugr AliasDefn or AliasDecl node";
    let assemblyFormat = "($sym_visibility^)? $extensions``$sym_name``(`,`$constraint^)? (`=` $aliasee^)? attr-dict";
    let description = [{
        A `type_alias` op represents an `AliasDefn` node if and only id `aliasee` is present.
        Otherwise it represents an `AliasDecl` node.

        TODO: Require that aleasee implements `HugrTypeInterface`
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility,
        OptionalAttr<TypeAttr>:$aliasee,
        Hugr_ExtensionSetAttr:$extensions,
        DefaultValuedAttr<Hugr_TypeConstraintAttr, "TypeConstraint::Equatable">:$constraint
    );

    let hasVerifier = 1;
    let extraClassDeclaration = [{
        bool isDeclaration() { return getAliasee().has_value(); }
    }];
}

def TailLoopPredicateType : Type<
   And<[PredicateType.predicate, CPred<"llvm::cast<SumType>($_self).getTypes().size() == 2">]>,
   "A predicate with two alternatives",
   PredicateType.cppClassName>;

class PassthroughMatches<string summary> : RangedTypesMatchWith<summary, "passthrough_inputs", "passthrough_outputs", "$_self">;

def Hugr_TailLoopOp : Hugr_Op<"tailloop",
    [ SingleBlock
    , PassthroughMatches<"TailLoop passthrough operands">
    , AttrSizedOperandSegments, AttrSizedResultSegments
    , HugrMemoryOp
    , OpAsmOpInterface
    , HugrSymbolUserOp
    ]> {
    let summary = "Hugr TailLoop op";
    let assemblyFormat = "$predicate_type (`(`$inputs^`:`type($inputs)`)`)? (`passthrough` `(`$passthrough_inputs^`:`type($passthrough_inputs)`)`)?custom<TailLoopOpOutputTypes>(ref(type($passthrough_inputs)),type($outputs),type($passthrough_outputs)) attr-dict-with-keyword ($body^)?";
    let description = [{
        Represents a `hugr::ops::controlflow::TailLoop` node.

        TODO implement `RegionControlFlowOpInterface`
        TODO verify input/output types of region and type of predicate all match
    }];

    let arguments = (ins
        Variadic<HugrTypeInterface>:$inputs,
        Variadic<HugrTypeInterface>:$passthrough_inputs,
        TypeAttrOf<TailLoopPredicateType>:$predicate_type
    );
    let results = (outs
        Variadic<HugrTypeInterface>:$outputs,
        Variadic<HugrTypeInterface>:$passthrough_outputs
    );
    let regions = (region DataflowGraphRegion:$body);

    let extraClassDeclaration = [{
        static mlir::StringRef getDefaultDialect() { return "hugr"; }
    }];
}

def Hugr_ConditionalOp : Hugr_Op<"conditional",
    [ HugrMemoryOp
    , OpAsmOpInterface
    , HugrSymbolUserOp
    ]> {
    let summary = "Hugr Conditional op";
    let assemblyFormat = "`(`operands`:`type(operands)`)` `->`type($outputs) attr-dict-with-keyword $cases";
    let description = [{
        Represents a `hugr::ops::controlflow::Conditional` node.

        TODO implement `RegionControlFlowOpInterface`
    }];

    let arguments = (ins
        PredicateType:$predicate, // TODO should be non-empty
        Variadic<HugrTypeInterface>:$other_inputs
    );
    let results = (outs
        Variadic<HugrTypeInterface>:$outputs
    );
    let regions = (region
        VariadicRegion<DataflowGraphRegion>:$cases
    );

    let hasRegionVerifier = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "mlir::TypeRange":$result_types, "mlir::Value":$predicate, "mlir::ValueRange":$other_inputs), [{
          $_state.addTypes(result_types);
          $_state.addOperands(predicate);
          $_state.addOperands(other_inputs);
          if(auto pred_type = llvm::dyn_cast<SumType>(predicate.getType())) {
            for(auto i = 0u; i < pred_type.numAlts(); ++i) { $_state.addRegion(); }
          }
        }]>
    ];

    let extraClassDeclaration = [{
        static mlir::StringRef getDefaultDialect() { return "hugr"; }
        SumType getPredicateType() { return llvm::cast<SumType>(getPredicate().getType()); }
    }];
}

def Hugr_LiftOp : Hugr_Op<"lift",
  [ HugrMemoryOp, HugrSymbolUserOp, Pure ]> {
    let summary = "A Hugr Lift op";
    let assemblyFormat = "$extensions $inputs`:` functional-type($inputs,$outputs) attr-dict";
    let description = [{
        Represents a set of `hugr::ops::leaf::LeafOp::Lift` Lift Nodes.

        The extension set `extensions` is unioned into the extensions of
        `inputs` to form `outputs`, elementwise.

        TODO Implement InferTypeOpInterface
              => We will be able to omit the types of outputs in the assembly format

        TODO Verify types match up
    }];

    let arguments = (ins
        Hugr_ExtensionSetAttr:$extensions,
        Variadic<HugrTypeInterface>:$inputs
    );
    let results = (outs
        Variadic<HugrTypeInterface>:$outputs
    );
}

// def Hugr_LiftGraphOp : Hugr_Op<"lift_graph",
//   [Pure]> {
//   let summary = "A Hugr LiftGraph op";
//   let assemblyFormat = "$extensions $input `:` type($input) `->` type($output) attr-dict";
//   let description = [{
//       TODO how is this represented in hugr?
//
//       TODO Implement InferTypeOpInterface
//             => We will be able to omit the type of output in the assembly format
//
//       TODO Verify that the types match up
//   }];
//   let arguments = (ins
//       Hugr_ExtensionSetAttr:$extensions,
//       Hugr_FunctionType:$input
//   );
//   let results = (outs
//       Hugr_FunctionType:$output
//   );
// }

def FuncStaticEdgeAttr : Attr<CPred<"llvm::isa<hugr_mlir::FunctionType>(llvm::cast<mlir::TypedAttr>($_self).getType())">, "a static edge to a function"> {
    let baseAttr = Hugr_StaticEdgeAttr;
    let returnType = Hugr_StaticEdgeAttr.returnType;
    let storageType = Hugr_StaticEdgeAttr.storageType;
    let valueType = Hugr_FunctionType;
    let convertFromStorage = Hugr_StaticEdgeAttr.convertFromStorage;
}

def Hugr_CallOp : Hugr_Op<"call",
  [ AttrSizedOperandSegments
  // RangedTypesMatchWith<"inputs match function type", "callee", "inputs", "llvm::cast<FunctionType>($_self).getArgumentTypes()">
  // , RangedTypesMatchWith<"outputs match function type", "callee", "outputs", "llvm::cast<FunctionType>($_self).getResultTypes()">
  , HugrMemoryOp
  , HugrSymbolUserOp
  // , InferTypeOpInterface
  ]> {
    let summary = "A Hugr Call op";
    let assemblyFormat = "custom<CallInputsOutputs>($callee_attr,type($callee_value),$callee_value,$inputs,type($inputs),type($outputs)) attr-dict";
    let description = [{
        Represents a `hugr::op::dataflow::Call` node.

        TODO Verift that extensions of callee, inputs, outputs must match

        TODO Implement InferTypeOpInterfcae
               => We will be able to omit the type of output in the assembly format

        TODO Implement CallOpInterface
    }];
    let arguments = (ins
        Optional<Hugr_FunctionType>:$callee_value,
        Variadic<HugrTypeInterface>:$inputs,
        OptionalAttr<FuncStaticEdgeAttr>:$callee_attr
    );
    let results = (outs
        Variadic<HugrTypeInterface>:$outputs
    );

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        FunctionType getFunctionType();
        // { return llvm::cast<FunctionType>(getCallee().getType()); }
    }];
}

def Hugr_OutputOp : Hugr_Op<"output",
  [Terminator,ReturnLike, HugrMemoryOp
  , HugrSymbolUserOp ]> {
    let summary = "A Hugr ouput node";
    let assemblyFormat = "($outputs^ `:` type($outputs))? attr-dict";
    let description = [{
        Represents a `hugr::ops::dataflow::Output` node.

        TODO Implement RegionControlFlowTerminatorOpInterface
    }];
    let arguments = (ins
        Variadic<HugrTypeInterface>:$outputs
    );
}

def Hugr_CfgOp : Hugr_Op<"cfg",
    [ OpAsmOpInterface, HugrMemoryOp, HugrSymbolUserOp
    // TODO RegionBranchOpInterface ,
    ]
    > {
    let summary = "A Hugr CFG";
    let assemblyFormat = "($inputs^)? `:` functional-type($inputs,$outputs) attr-dict-with-keyword( $body^)?";
    let description = [{
        Represents a `hugr::ops::controlflow::CFG` Node.

        Control Flow Edges within the body are represented by `hugr.switch` ops.

        TODO Implement RegionBranchOpInterface
    }];
    let arguments = (ins
        Variadic<HugrTypeInterface>:$inputs
    );
    let results = (outs
        Variadic<HugrTypeInterface>:$outputs
    );

    let regions = (region ControlFlowGraphRegion:$body);

    let hasRegionVerifier = 1;
    let extraClassDeclaration = [{
        static mlir::StringRef getDefaultDialect() { return "hugr"; }
    }];
}

def Hugr_SwitchOp : Hugr_Op<"switch", [HugrMemoryOp, HugrSymbolUserOp, Terminator]> {
    let summary = "A branching terminator for hugr.cfg body regions";
    let assemblyFormat = "operands`:` type(operands) $destinations attr-dict";
    let description = "TODO";
    let arguments = (ins
        PredicateType:$predicate,
        Variadic<HugrTypeInterface>:$other_inputs
    );
    let successors = (successor
        VariadicSuccessor<AnySuccessor>:$destinations
    );
    let hasVerifier = 1;
}

def Hugr_DfgOp : Hugr_Op<"dfg",
    [ SingleBlock
    , OpAsmOpInterface
    , HugrMemoryOp, HugrSymbolUserOp
    ]> {
    let summary = "A Hugr DFG";
    let assemblyFormat = "(`input` `extensions` $input_extensions^)? $inputs `:` functional-type($inputs, $outputs) attr-dict-with-keyword( $body^)?";
    let description = [{
        Represents a `hugr::ops::dataflow::DFG` Node.

        TODO Implement RegionBranchOpInterface
    }];
    let arguments = (ins
        OptionalAttr<Hugr_ExtensionSetAttr>:$input_extensions,
        Variadic<HugrTypeInterface>:$inputs
    );
    let results = (outs
        Variadic<HugrTypeInterface>:$outputs
    );

    let regions = (region DataflowGraphRegion:$body);

    let hasRegionVerifier = 1;
    let extraClassDeclaration = [{
        static mlir::StringRef getDefaultDialect() { return "hugr"; }
    }];

}


def Hugr_ExtensionOp : Hugr_Op<"ext_op",
    [HugrMemoryOp, HugrSymbolUserOp]> {
    let summary = "A node representing an arbitrary extension op node in Hugr.";
    let assemblyFormat = "$extensions``$hugr_opname $args `:` functional-type($args,$results) attr-dict";
    let description = [{
        An arbtrary extension op node in Hugr.

        Note that we will have an `mlir` extension with an `mlir_type` and `mlir_op` that can represent any mlir operation that satisfies:
        * No Regions
        * No Successors
        * All attributes satisfy constraints that guarantee they can round trip

        TODO: what constraints are there on node_weight?
        TODO: Should this C++ provide an interface to delegate to the hugr
        crate? For example to check that  the op is well formed according to the
        extension spec.
    }];

    let arguments = (ins
        Variadic<HugrTypeInterface>:$args,
        StrAttr:$hugr_opname,
        Hugr_ExtensionSetAttr:$extensions,
        OptionalAttr<DictionaryAttr>:$node_weight
    );

    let results = (outs
        Variadic<HugrTypeInterface>:$results
    );
}

def Hugr_PanicOp : Hugr_Op<"panic",
    [Terminator]> {
    let summary = "A node representing a call to panic";
    let assemblyFormat = "$errno $message attr-dict";
    let description = [{
        This is not a primitive in Hugr, but we expect this to be useful here as a terminator.
    }];
    let arguments = (ins
        Index:$errno,
        StrAttr:$message
    );
}

def Hugr_ConstOp : Hugr_Op<"const",
    [Symbol, HugrSymbolUserOp]> {
    let summary = "A Hugr Const node";
    let assemblyFormat = "$sym_name `:` $type (`=` $value^)? attr-dict";
    let description = [{
        Represents a `hugr::ops::constant::Const` node.

        TODO: define acceptable types(HugrTypeInterface) and acceptable
        attributes for value.
    }];
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        DefaultValuedStrAttr<StrAttr,"private">:$sym_visibility,
        TypeAttr:$type,
        OptionalAttr<TypedAttrInterface>:$value
    );
}

def Hugr_LoadConstantOp : Hugr_Op<"load_constant",
    [ ConstantLike
    , DeclareOpInterfaceMethods<InferTypeOpInterface>
    , HugrSymbolUserOp
    ]> {
    let summary = "A Hugr LoadConstant node";
    let assemblyFormat = "custom<StaticEdge>($const_ref) attr-dict";
    let description = [{
        Represents a `hugr::ops::dataflow::LoadConstant` node.

        TODO verify getResult().getType() == getConstRef().getType()
        TODO define acceptable target ops for const_ref
        TODO verify const_ref in SymbolUserOpInterface
        TODO Implement InferTypeOpInterface
    }];

    let arguments = (ins
        Hugr_StaticEdgeAttr:$const_ref
    );

    let results = (outs
        HugrTypeInterface:$result
    );
}

def Hugr_MakeTupleOp : Hugr_Op<"make_tuple",
    [ Pure
    , DeclareOpInterfaceMethods<InferTypeOpInterface>
    , HugrMemoryOp
    , HugrSymbolUserOp
    // , InferTypeOpInterface
    ]> {
    let summary = "A Hugr LeafOp::MakeTuple node";
    let assemblyFormat = "`(`($inputs^ `:` type($inputs))?`)` attr-dict";
    let description = [{
        Represents a `hugr::ops::leaf::LeafOp::MakeTuple` node
    }];

    let arguments = (ins
        Variadic<HugrTypeInterface>:$inputs
    );
    let results = (outs
        Builtin_Tuple:$output
    );

    let hasVerifier = 1;
    let hasFolder = 1;
}

def Hugr_UnpackTupleOp : Hugr_Op<"unpack_tuple",
    [ Pure
    , DeclareOpInterfaceMethods<InferTypeOpInterface>
    , HugrMemoryOp
    , HugrSymbolUserOp
    ]> {
    let summary = "A Hugr LeafOp::UnpackTuple node";
    let assemblyFormat = "$input `:` type($input) attr-dict";
    let description = [{
        Represents a `hugr::ops::leaf::LeafOp::UnpackTuple` node.
    }];

    let arguments = (ins
        Builtin_Tuple:$input
    );

    let results = (outs
        Variadic<HugrTypeInterface>:$outputs
    );

    let hasVerifier = 1;
    let hasFolder = 1;
}

def Hugr_TagOp : Hugr_Op<"tag",
    [ Pure, HugrMemoryOp
    , HugrSymbolUserOp
    // , InferTypeOpInterface
    ]> {
    let summary = "A Hugr LeafOp::Tag node";
    let assemblyFormat = "$tag $input `:` type($input) `->` type($output) attr-dict";
    let description = [{
        Represents a `hugr::ops::leaf::LeafOp::Tag` node

        TODO verify input/output types
    }];

    let arguments = (ins
        IndexAttr:$tag,
        HugrTypeInterface:$input
    );

    let results = (outs
        Hugr_SumType:$output
    );

    let hasVerifier = 1;
    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "mlir::Type":$type, "mlir::Value":$input, "unsigned":$tag), [{
          $_state.addTypes(type);
          $_state.addOperands(input);
          $_state.getOrAddProperties<Properties>().tag = $_builder.getIndexAttr(tag);
        }]>
    ];
}

def Hugr_ReadTagOp : Hugr_Op<"read_tag",
   [ Pure, HugrMemoryOp, HugrSymbolUserOp ]> {
    let summary = "An implementation op to read the tag of a sum";
    let assemblyFormat = "$input `:` type($input) attr-dict";
    let description = [{
        TODO
    }];

    let arguments = (ins
        Hugr_SumType:$input
    );

    let results = (outs
        Index:$output
    );

    let hasFolder = 1;
}

def Hugr_ReadVariantOp : Hugr_Op<"read_variant",
   [ Pure, HugrMemoryOp, HugrSymbolUserOp,
   DeclareOpInterfaceMethods<InferTypeOpInterface>
   ]> {
    let summary = "An implementation op to read the tag of a sum";
    let assemblyFormat = "$tag $input `:` type($input) attr-dict";
    let description = [{
        TODO
    }];

    let arguments = (ins
        IndexAttr:$tag,
        Hugr_SumType:$input
    );

    let results = (outs
        HugrTypeInterface:$output
    );

    let builders = [
        OpBuilder<(ins "mlir::Type":$type, "::mlir::Value":$input, "unsigned":$tag), [{
          $_state.addTypes(type);
          $_state.addOperands(input);
          $_state.getOrAddProperties<Properties>().tag = $_builder.getIndexAttr(tag);
        }]>
    ];

    let hasFolder = 1;
}


def Hugr_IdentityOp : Hugr_Op<"identity",
    [ HugrMemoryOp
    , SameOperandsAndResultType
    , HugrSymbolUserOp
    ]> {
    let summary = "A Hugr Identity op";
    let assemblyFormat = "$inputs `:` type($inputs) attr-dict";
    let description = [{
        Represents a `hugr::ops::leaf::Noop`
    }];
    let arguments = (ins
        Variadic<HugrTypeInterface>:$inputs
    );
    let results = (outs
        Variadic<HugrTypeInterface>:$outputs
    );
}


#endif
