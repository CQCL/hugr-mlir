#include "hugr-mlir/IR/HugrOps.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "hugr-mlir/IR/HugrOps.cpp.inc"

/////////////////////////////////////////////////////////////////////////////
//  FuncOp
/////////////////////////////////////////////////////////////////////////////
mlir::ArrayRef<mlir::Type> hugr_mlir::FuncOp::getResultTypes() {
  return getFunctionType().getResultTypes();
}

mlir::ArrayRef<mlir::Type> hugr_mlir::FuncOp::getArgumentTypes() {
  return getFunctionType().getArgumentTypes();
}

// Adapted from mlir/lib/IR/FunctionImplementation.cpp
mlir::ParseResult hugr_mlir::FuncOp::parse(
    mlir::OpAsmParser& parser, mlir::OperationState& result) {
  auto context = parser.getContext();
  mlir::OperationName func_opname(FuncOp::getOperationName(), context);
  auto sym_name_attr_name = FuncOp::getSymNameAttrName(func_opname);
  auto sym_visibility_attr_name = FuncOp::getSymVisibilityAttrName(func_opname);
  auto function_type_attr_name = FuncOp::getFunctionTypeAttrName(func_opname);
  auto arg_attrs_attr_name = FuncOp::getArgAttrsAttrName(func_opname);
  auto res_attrs_attr_name = FuncOp::getResAttrsAttrName(func_opname);

  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  ExtensionSetAttr extensions;
  mlir::StringAttr name_attr;
  mlir::SmallVector<mlir::OpAsmParser::Argument> args;
  mlir::SmallVector<mlir::Type> result_types;
  mlir::SmallVector<mlir::DictionaryAttr> result_attrs;

  bool is_variadic;
  if (parser.parseSymbolName(
          name_attr, sym_name_attr_name, result.attributes)) {
    return mlir::failure();
  }

  // Parse the function signature.
  mlir::SMLoc signatureLocation = parser.getCurrentLocation();
  if (parser.parseCustomAttributeWithFallback<ExtensionSetAttr>(extensions) ||
      mlir::function_interface_impl::parseFunctionSignature(
          parser, false,  // don't allow variadic
          args, is_variadic, result_types, result_attrs)) {
    return mlir::failure();
  }

  mlir::SmallVector<mlir::Type> arg_types;
  llvm::transform(
      args, std::back_inserter(arg_types), [](auto x) { return x.type; });

  auto get_err = [&parser, &signatureLocation]() -> mlir::InFlightDiagnostic {
    return std::move(
        parser.emitError(signatureLocation)
        << "failed to construct function type:");
  };
  // mlir::FunctionType does not have a verifier, so we don't  call getChecked
  auto inner_func_type = ::mlir::FunctionType::get(
      context, mlir::ArrayRef(arg_types), mlir::ArrayRef(result_types));
  auto func_type =
      FunctionType::getChecked(get_err, context, extensions, inner_func_type);
  if (!func_type) {
    return mlir::failure();
  }
  result.addAttribute(function_type_attr_name, mlir::TypeAttr::get(func_type));

  mlir::NamedAttrList parsedAttributes;
  auto attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes)) {
    return mlir::failure();
  }

  // we don't disallow arg_attrs, or res_attrs. Just copying
  // FunctionImplementation.cpp here
  for (auto disallowed :
       {sym_name_attr_name, sym_visibility_attr_name,
        function_type_attr_name}) {
    if (parsedAttributes.get(disallowed)) {
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
    }
  }
  result.attributes.append(parsedAttributes);

  // Add the attributes to the function arguments.
  assert(result_attrs.size() == result_types.size());
  mlir::function_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, args, result_attrs, arg_attrs_attr_name,
      res_attrs_attr_name);

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto* body = result.addRegion();
  auto loc = parser.getCurrentLocation();
  auto parseResult = parser.parseOptionalRegion(
      *body, args,
      /*enableNameShadowing=*/false);

  if (parseResult.has_value()) {
    if (mlir::failed(*parseResult)) return mlir::failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expectednon-empty function body");
  }
  return mlir::success();
}

void hugr_mlir::FuncOp::print(mlir::OpAsmPrinter& p) {
  p << ' ';
  auto visibility = getSymVisibility();
  if (visibility != "private") {
    p << visibility << ' ';
  }
  p.printSymbolName(getSymName());
  p.printStrippedAttrOrType(getExtensionSet());

  mlir::ArrayRef<mlir::Type> argTypes = getArgumentTypes();
  mlir::ArrayRef<mlir::Type> resultTypes = getResultTypes();
  auto foi = llvm::cast<mlir::FunctionOpInterface>(getOperation());
  mlir::function_interface_impl::printFunctionSignature(
      p, foi, argTypes, false, resultTypes);
  mlir::function_interface_impl::printFunctionAttributes(
      p, foi,
      {getSymVisibilityAttrName().getValue(),
       getFunctionTypeAttrName().getValue(), getArgAttrsAttrName().getValue(),
       getResAttrsAttrName().getValue()});

  if (!getBody().empty()) {
    p << ' ';
    p.printRegion(
        getBody(), /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/true);
  }
}

/////////////////////////////////////////////////////////////////////////////
//  CallOp
/////////////////////////////////////////////////////////////////////////////
mlir::ParseResult hugr_mlir::parseCallInputsOutputs(
    ::mlir::OpAsmParser& parser, StaticEdgeAttr& callee,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& inputs,
    mlir::SmallVectorImpl<mlir::Type>& inputTypes,
    mlir::SmallVectorImpl<mlir::Type>& outputTypes) {
  mlir::SymbolRefAttr sym;

  if (parser.parseAttribute(sym)) {
    return mlir::failure();
  }
  auto t = FunctionType::parse(parser);
  auto func_type = llvm::dyn_cast_if_present<FunctionType>(t);
  if (!func_type) {
    return parser.emitError(
               parser.getCurrentLocation(), "Callee is not a !hugr.function:")
           << sym << ": " << t;
  }
  callee = parser.getBuilder().getAttr<StaticEdgeAttr>(func_type, sym);

  if (parser.parseOperandList(inputs, func_type.getArgumentTypes().size())) {
    return mlir::failure();
  }
  llvm::copy(func_type.getArgumentTypes(), std::back_inserter(inputTypes));
  llvm::copy(func_type.getResultTypes(), std::back_inserter(outputTypes));
  return mlir::success();
}

void hugr_mlir::printCallInputsOutputs(
    ::mlir::OpAsmPrinter& printer, CallOp, StaticEdgeAttr callee,
    mlir::OperandRange inputs, mlir::TypeRange, mlir::TypeRange) {
  printer << callee.getRef() << " ";
  printer.printStrippedAttrOrType<FunctionType>(
      llvm::cast<FunctionType>(callee.getType()));
  printer << " ";
  printer.printOperands(inputs);
}

/////////////////////////////////////////////////////////////////////////////
//  LoadConstantOp
/////////////////////////////////////////////////////////////////////////////
mlir::ParseResult hugr_mlir::parseStaticEdge(
    ::mlir::OpAsmParser& parser, StaticEdgeAttr& result) {
  mlir::SymbolRefAttr sym;

  auto start_loc = parser.getCurrentLocation();

  if (parser.parseAttribute(sym)) {
    return mlir::failure();
  }

  mlir::Type type;
  if (parser.parseColonType(type)) {
    return mlir::failure();
  }
  auto err_fn = [&parser, &start_loc]() -> mlir::InFlightDiagnostic {
    return parser.emitError(start_loc, "Failed to parse static edge");
  };
  StaticEdgeAttr attr =
      StaticEdgeAttr::getChecked(err_fn, parser.getContext(), type, sym);
  if (!attr) {
    return mlir::failure();
  }
  result = attr;
  return mlir::success();
}

void hugr_mlir::printStaticEdge(
    ::mlir::OpAsmPrinter& printer, mlir::Operation*,
    StaticEdgeAttr static_edge) {
  printer << static_edge.getRef() << " : " << static_edge.getType();
}

mlir::LogicalResult hugr_mlir::LoadConstantOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  LoadConstantOpAdaptor op(operands, attributes, properties, regions);
  auto const_ref = op.getConstRef();
  if (!const_ref || !const_ref.getType()) {
    return mlir::failure();
  }
  inferredReturnTypes.push_back(const_ref.getType());
  return mlir::success();
}

/////////////////////////////////////////////////////////////////////////////
//  TailLoopOp
/////////////////////////////////////////////////////////////////////////////
mlir::ParseResult hugr_mlir::parseTailLoopOpOutputTypes(
    ::mlir::OpAsmParser& parser,
    mlir::SmallVectorImpl<mlir::Type> const& passthrough_input_types,
    mlir::SmallVectorImpl<mlir::Type>& output_types,
    mlir::SmallVectorImpl<mlir::Type>& passthrough_output_types) {
  if (parser.parseArrowTypeList(output_types)) {
    return mlir::failure();
  }
  llvm::copy(
      passthrough_input_types, std::back_inserter(passthrough_output_types));
  return mlir::success();
}

void hugr_mlir::printTailLoopOpOutputTypes(
    ::mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::TypeRange,
    mlir::TypeRange output_types, mlir::TypeRange) {
  printer << "-> (" << output_types << ")";
}

/////////////////////////////////////////////////////////////////////////////
//  ConditionalOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::ConditionalOp::verify() {
  auto n = getPredicateType().numAlts();
  if (getNumRegions() > 0 && n != getNumRegions()) {
    return emitOpError("Number of regions does not match predicate type '")
           << getPredicateType() << "' with " << getNumRegions() << " regions";
  }
  return mlir::success();
}

mlir::LogicalResult hugr_mlir::ConditionalOp::verifyRegions() {
  for (auto* r : getRegions()) {
    if (r->empty()) {
      continue;
    }
    auto alt_types = llvm::cast<mlir::TupleType>(
        getPredicateType().getAltType(r->getRegionNumber()));

    mlir::SmallVector<mlir::Type> expected_types(alt_types.getTypes());
    llvm::copy(getOtherInputs().getTypes(), std::back_inserter(expected_types));
    if (r->getArgumentTypes() != expected_types) {
      return emitOpError("Region ")
             << r->getRegionNumber()
             << " has unexpected region argument types. Expected ("
             << expected_types << "), found " << r->getArgumentTypes();
    }
  }
  return mlir::success();
  // TODO verify output types of regions
}

/////////////////////////////////////////////////////////////////////////////
//  MakeTupleOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::MakeTupleOp::verify() {
  if (mlir::TypeRange(getOutput().getType().getTypes()) !=
      getInputs().getTypes()) {
    return emitOpError("Inputs and output types do not match: Inputs=")
           << getInputs().getTypes()
           << ", expected output type: " << getOutput().getType().getTypes();
  }
  return mlir::success();
};

mlir::LogicalResult hugr_mlir::MakeTupleOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(
      mlir::TupleType::get(context, operands.getTypes()));
  return mlir::success();
}

/////////////////////////////////////////////////////////////////////////////
//  UnpackTupleOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::UnpackTupleOp::verify() {
  if (mlir::TypeRange(getInput().getType().getTypes()) !=
      getOutputs().getTypes()) {
    return emitOpError("Input and outputs types do not match: Input=")
           << getInput().getType().getTypes()
           << ", expected output types: " << getOutputs().getTypes();
  }
  return mlir::success();
};

mlir::LogicalResult hugr_mlir::UnpackTupleOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  UnpackTupleOpAdaptor adaptor(operands, attributes, properties, regions);
  if (!adaptor.getInput()) {
    return mlir::failure();
  }
  if (auto t = llvm::dyn_cast<mlir::TupleType>(adaptor.getInput().getType())) {
    llvm::copy(t.getTypes(), std::back_inserter(inferredReturnTypes));
  } else {
    return mlir::failure();
  }
  return mlir::success();
}

/////////////////////////////////////////////////////////////////////////////
//  CfgOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::CfgOp::verifyRegions() {
  llvm::SmallVector<OutputOp> output_ops;
  for (auto& block : getBody()) {
    assert(
        !block.empty() &&
        "Region has already been verified by ControlFlowGraphRegion");
    if (auto o = llvm::dyn_cast<OutputOp>(block.back())) {
      output_ops.push_back(o);
    }
  }

  auto output_types = getOutputs().getTypes();
  auto it = llvm::find_if(output_ops, [&output_types](OutputOp o) {
    return !areTypesCompatible(output_types, o.getOutputs().getTypes());
  });
  if (it != output_ops.end()) {
    auto ifd =
        emitOpError("output op in body has incompatible types: Expected (")
        << output_types << "), found (" << it->getOutputs().getTypes() << ")";
    ifd.attachNote(it->getLoc()) << *it;
    return ifd;
  }

  return mlir::success();
}
/////////////////////////////////////////////////////////////////////////////
//  SwitchOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::SwitchOp::verify() {
  auto pred_type = llvm::cast<SumType>(getPredicate().getType());
  if (pred_type.numAlts() == 0) {
    return emitOpError("Predicate type must have more than zero alts");
  }
  if (pred_type.numAlts() != getDestinations().size()) {
    return emitOpError("Predicate type has the wrong number of alts. ")
           << pred_type << ", with " << getDestinations().size() << " alts";
  }
  for (auto [i, x] : llvm::enumerate(
           llvm::zip_equal(pred_type.getTypes(), getDestinations()))) {
    auto [t, dest] = x;
    auto tuple_type = llvm::cast<mlir::TupleType>(t);
    mlir::SmallVector<mlir::Type> types_for_block{tuple_type.getTypes()};
    llvm::copy(
        getOtherInputs().getTypes(), std::back_inserter(types_for_block));
    if (!areTypesCompatible(types_for_block, dest->getArgumentTypes())) {
      return emitOpError("type mismatch for successor ")
             << i << ". Block expects (" << dest->getArgumentTypes()
             << "), but op provides (" << types_for_block << ")";
    }
  }

  return mlir::success();
}

/////////////////////////////////////////////////////////////////////////////
//  DfgOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::DfgOp::verifyRegions() {
  if (getBody().empty()) {
    return mlir::success();
  }
  {
    mlir::SmallVector<mlir::Type> expected_region_types;
    if (auto input_extensions = getInputExtensions()) {
      llvm::transform(
          getInputs().getTypes(), std::back_inserter(expected_region_types),
          [&input_extensions](mlir::Type t) {
            return ExtendedType::get(llvm::cast<HugrTypeInterface>(t))
                .removeExtensions(*input_extensions);
          });
    } else {
      llvm::copy(
          getInputs().getTypes(), std::back_inserter(expected_region_types));
    }

    if (!areTypesCompatible(
            getBody().getArgumentTypes(), expected_region_types)) {
      return emitOpError("Body argument type mismatch: Expected (")
             << expected_region_types << "), found ("
             << getBody().getArgumentTypes() << ")";
    }
  }

  assert(
      !getBody().front().empty() &&
      "Region has already been verified by DataflowGraphRegion");
  if (auto output_op = llvm::dyn_cast<OutputOp>(getBody().front().front())) {
    if (!areTypesCompatible(
            output_op.getOutputs().getTypes(), getOutputs().getTypes())) {
      auto ifd = emitOpError("Body output type mismatch: Expected (")
                 << getOutputs().getTypes() << "), found ("
                 << output_op.getOutputs().getTypes() << ")";
      ifd.attachNote(output_op.getLoc()) << output_op;
      return ifd;
    }
  }

  return mlir::success();
}

/////////////////////////////////////////////////////////////////////////////
// Miscilaneous free functions
/////////////////////////////////////////////////////////////////////////////
void hugr_mlir::getHugrTypeMemoryEffects(
    mlir::Operation* op,
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>&
        effects) {
  auto add_linear_effect = [&effects](
                               mlir::Value v, mlir::MemoryEffects::Effect* e) {
    if (auto htt = llvm::dyn_cast<HugrTypeInterface>(v.getType())) {
      if (htt.getConstraint() == TypeConstraint::Linear) {
        effects.push_back(
            mlir::SideEffects::EffectInstance(e, v, LinearityResource::get()));
      }
    }
  };

  for (auto operand : op->getOperands()) {
    add_linear_effect(operand, mlir::MemoryEffects::Free::get());
  }

  for (auto result : op->getResults()) {
    add_linear_effect(result, mlir::MemoryEffects::Allocate::get());
  }
}

bool hugr_mlir::isDataflowGraphRegion(mlir::Region& region) {
  if (region.empty()) {
    return true;
  }  // An empty region is always ok
  if (!region.hasOneBlock() || region.front().empty()) {
    return false;
  }

  // TODO we should likely accept at least LLVM::UnreachableOp
  return llvm::isa<OutputOp, PanicOp>(region.front().back());
}

bool hugr_mlir::isControlFlowGraphRegion(mlir::Region& region) {
  for (auto& block : region) {
    // TODO LLVM::UnreachableOp
    if (block.empty() ||
        !llvm::isa<
            OutputOp, PanicOp, SwitchOp, mlir::cf::CondBranchOp,
            mlir::cf::BranchOp, mlir::cf::SwitchOp>(block.back())) {
      return false;
    }
  }
  return true;
}

void hugr_mlir::HugrDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "hugr-mlir/IR/HugrOps.cpp.inc"
      >();
}
