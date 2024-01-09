#include "hugr-mlir/IR/HugrOps.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "hugr-mlir/IR/HugrOps.cpp.inc"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "hugr-ops"

/////////////////////////////////////////////////////////////////////////////
//  ModuleOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::ModuleOp::verifySymbolUses(
    ::mlir::SymbolTableCollection& symbolTable) {
  HugrSymbolMap map;
  std::optional<mlir::InFlightDiagnostic> mb_ifd;
  auto ifd = [&](mlir::Twine t) -> mlir::InFlightDiagnostic& {
    if (!mb_ifd) {
      mb_ifd.emplace(emitError(t));
    }
    return *mb_ifd;
  };
  getOperation()->walk([&](mlir::Operation* op) {
    if (op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      return mlir::WalkResult::skip();
    }
    if (auto sym = llvm::dyn_cast<mlir::SymbolOpInterface>(op)) {
      auto r = map.insert(std::make_pair(
          mlir::FlatSymbolRefAttr::get(sym.getNameAttr()), sym.getOperation()));
      if (!r.second) {
        auto& e = ifd("hugr.module contains duplicate definitions for symbol:");
        e.attachNote(sym->getLoc()) << sym->getName();
        e.attachNote(r.first->second->getLoc()) << r.first->second->getName();
        return mlir::WalkResult::interrupt();
      };
    }
    return mlir::WalkResult::advance();
  });
  for (auto const& [k, v] : map) {
    if (mlir::failed(verifyHugrSymbolUses(v, map))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

/////////////////////////////////////////////////////////////////////////////
//  FuncOp
/////////////////////////////////////////////////////////////////////////////

hugr_mlir::StaticEdgeAttr hugr_mlir::FuncOp::getStaticEdgeAttr() {
  mlir::OpBuilder builder(getContext());
  return builder.getAttr<StaticEdgeAttr>(
      getFunctionType(),
      builder.getAttr<mlir::SymbolRefAttr>(getSymNameAttr()));
}

mlir::ArrayRef<mlir::Type> hugr_mlir::FuncOp::getResultTypes() {
  return getFunctionType().getResultTypes();
}

mlir::ArrayRef<mlir::Type> hugr_mlir::FuncOp::getArgumentTypes() {
  return getFunctionType().getArgumentTypes();
}

mlir::Region* hugr_mlir::FuncOp::getCallableRegion() { return &getBody(); }

mlir::LogicalResult hugr_mlir::FuncOp::verifyBody() {
  using namespace mlir;
  if (isExternal()) return success();
  SmallVector<Type> fnInputTypes{getCaptures().getTypes()};
  llvm::copy(getArgumentTypes(), std::back_inserter(fnInputTypes));
  Block& entryBlock = getBody().front();

  unsigned numArguments = fnInputTypes.size();
  if (entryBlock.getNumArguments() != numArguments)
    return emitOpError("entry block must have ")
           << numArguments
           << " arguments to match function captures and signature";

  for (unsigned i = 0, e = fnInputTypes.size(); i != e; ++i) {
    Type argType = entryBlock.getArgument(i).getType();
    if (fnInputTypes[i] != argType) {
      return emitOpError("type of entry block argument #")
             << i << '(' << argType
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';
    }
  }

  return success();
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
  mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand> captures;
  mlir::SmallVector<mlir::OpAsmParser::Argument> args;
  mlir::SmallVector<mlir::Type> result_types;
  mlir::SmallVector<mlir::DictionaryAttr> result_attrs;

  bool is_variadic;
  if (parser.parseSymbolName(
          name_attr, sym_name_attr_name, result.attributes)) {
    return mlir::failure();
  }

  if (parser.parseOperandList(captures)) {
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

  if (args.size() < captures.size()) {
    return parser.emitError(signatureLocation) << "more captures than args";
  }
  mlir::ArrayRef<mlir::OpAsmParser::Argument> capture_args =
      mlir::ArrayRef(args).take_front(captures.size());
  mlir::ArrayRef<mlir::OpAsmParser::Argument> func_args =
      mlir::ArrayRef(args).drop_front(captures.size());
  mlir::SmallVector<mlir::Type> arg_types;
  llvm::transform(
      func_args, std::back_inserter(arg_types), [](auto x) { return x.type; });

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
  mlir::SmallVector<mlir::Value> captures_vals;
  for (auto [c, a] : llvm::zip_equal(captures, capture_args)) {
    if (parser.resolveOperand(c, a.type, captures_vals)) {
      return mlir::failure();
    }
  }
  result.addOperands(captures_vals);
  return mlir::success();
}

void hugr_mlir::FuncOp::print(mlir::OpAsmPrinter& p) {
  p << ' ';
  auto visibility = getSymVisibility();
  if (visibility != "private") {
    p << visibility << ' ';
  }
  p.printSymbolName(getSymName());
  p.printOperands(getCaptures());

  p.printStrippedAttrOrType(getExtensionSet());

  mlir::SmallVector<mlir::Type> argTypes{getCaptures().getTypes()};
  llvm::copy(getArgumentTypes(), std::back_inserter(argTypes));
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
    ::mlir::OpAsmParser& parser, StaticEdgeAttr& callee_attr,
    mlir::Type& callee_value_type,
    std::optional<mlir::OpAsmParser::UnresolvedOperand>& callee_value,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>& inputs,
    mlir::SmallVectorImpl<mlir::Type>& inputTypes,
    mlir::SmallVectorImpl<mlir::Type>& outputTypes) {
  mlir::SymbolRefAttr sym;
  mlir::OpAsmParser::UnresolvedOperand mb_val;
  auto val_parse_result = parser.parseOptionalOperand(mb_val);
  if (val_parse_result.has_value()) {
    if (*val_parse_result) {
      return mlir::failure();
    }
    callee_value = mb_val;
  } else {
    if (parser.parseAttribute(sym)) {
      return mlir::failure();
    }
  }

  auto func_type =
      llvm::dyn_cast_if_present<FunctionType>(FunctionType::parse(parser));
  if (!func_type) {
    return parser.emitError(parser.getCurrentLocation(), "no callee type");
  }
  if (sym) {
    callee_attr = parser.getBuilder().getAttr<StaticEdgeAttr>(func_type, sym);
    callee_value_type = nullptr;
    callee_value = std::nullopt;
  } else {
    callee_value_type = func_type;
    callee_attr = nullptr;
  }
  assert(
      ((callee_attr && !callee_value_type && !callee_value) ||
       (!callee_attr && callee_value_type && callee_value)) &&
      "callee is attribute xor valur");

  if (func_type.getArgumentTypes().size() > 0 &&
      parser.parseOperandList(inputs, func_type.getArgumentTypes().size())) {
    return mlir::failure();
  }

  llvm::copy(func_type.getArgumentTypes(), std::back_inserter(inputTypes));
  llvm::copy(func_type.getResultTypes(), std::back_inserter(outputTypes));
  return mlir::success();
}

void hugr_mlir::printCallInputsOutputs(
    ::mlir::OpAsmPrinter& printer, CallOp op, StaticEdgeAttr, mlir::Type,
    std::optional<mlir::Value>, mlir::OperandRange, mlir::TypeRange,
    mlir::TypeRange) {
  mlir::Type func_type;

  if (auto callee_attr = op.getCalleeAttrAttr()) {
    printer << callee_attr.getRef() << " ";
    func_type = callee_attr.getType();
  } else {
    auto callee = op.getCalleeValue();
    assert(callee && "no callee attr means there must be a callee value");
    printer << op.getCalleeValue() << " ";
    func_type = callee.getType();
  }

  printer.printStrippedAttrOrType<FunctionType>(
      llvm::cast<FunctionType>(func_type));
  printer << " ";
  printer.printOperands(op.getInputs());
}

mlir::LogicalResult hugr_mlir::CallOp::verify() {
  if (getCalleeAttr().has_value() == (getCalleeValue() != nullptr)) {
    return emitOpError(
               "call must have exactly one of callee_attr and "
               "callee_value:callee_value=")
           << getCalleeValue() << ",callee_attr=" << getCalleeAttrAttr();
  }
  if (!areTypesCompatible(
          getFunctionType().getArgumentTypes(), getInputs().getTypes())) {
    return emitOpError("Arguments Type mismatch. Expected from callee:(")
           << getFunctionType().getArgumentTypes() << "), but found ("
           << getInputs().getTypes() << ")";
  }
  if (!areTypesCompatible(
          getFunctionType().getResultTypes(), getOutputs().getTypes())) {
    return emitOpError("Results Type mismatch. Expected from callee:(")
           << getFunctionType().getResultTypes() << "), but found ("
           << getOutputs().getTypes() << ")";
  }
  return mlir::success();
}

hugr_mlir::FunctionType hugr_mlir::CallOp::getFunctionType() {
  // INVARIANT: ODS verifiers have passed
  assert(
      getCalleeAttr().has_value() != (getCalleeValue() != nullptr) &&
      "callee_attr xor callee_value");

  // ODS declarations guarantee these casts will succeed
  if (auto attr = getCalleeAttrAttr()) {
    return llvm::cast<FunctionType>(attr.getType());
  } else {
    return llvm::cast<FunctionType>(getCalleeValue().getType());
  }
}

mlir::CallInterfaceCallable hugr_mlir::CallOp::getCallableForCallee() {
  if (auto attr = getCalleeAttrAttr()) {
    return attr.getRef();
  }
  return getCalleeValue();
}

void hugr_mlir::CallOp::setCalleeFromCallable(
    ::mlir::CallInterfaceCallable callee) {
  llvm::TypeSwitch<::mlir::CallInterfaceCallable>(callee)
      .Case([&](mlir::Value v) {
        setCalleeAttrAttr(nullptr);
        getCalleeValueMutable().assign(v);
      })
      .Case([&](mlir::SymbolRefAttr a) {
        getCalleeValueMutable().clear();
        // TODO this function type gets no extensions, dodgy. perhaps panic
        // here?
        auto ft = FunctionType::get(
            ExtensionSetAttr::get(getContext()),
            mlir::FunctionType::get(
                getContext(), getInputs().getType(), getOutputs().getType()));
        setCalleeAttrAttr(::hugr_mlir::StaticEdgeAttr::get(ft, a));
      });
}

mlir::Operation::operand_range hugr_mlir::CallOp::getArgOperands() {
  return getInputs();
}
mlir::MutableOperandRange hugr_mlir::CallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

mlir::LogicalResult hugr_mlir::CallOp::verifyHugrSymbolUses(
    HugrSymbolMap const& map) {
  auto callee_attr = getCalleeAttrAttr();
  if (!callee_attr) {
    return mlir::success();
  }

  auto callee_op = map.lookup(callee_attr.getRef());
  if (!callee_op) {
    return emitOpError("Unknown symbol: ") << callee_attr.getRef();
  }

  auto callee_func = llvm::dyn_cast<FuncOp>(callee_op);
  if (!callee_func) {
    return emitOpError("Symbol References op of type: ")
           << callee_op->getName() << ", expected "
           << FuncOp::getOperationName();
  }

  if (callee_func.getFunctionType() != callee_attr.getType()) {
    return emitOpError("Callee has type: ")
           << callee_func.getFunctionType()
           << ", expected: " << callee_attr.getType();
  }

  return mlir::success();
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

mlir::OpFoldResult hugr_mlir::MakeTupleOp::fold(FoldAdaptor adaptor) {
  auto mb_poison = llvm::find_if(adaptor.getInputs(), [](auto a) {
    return llvm::isa_and_present<mlir::ub::PoisonAttrInterface>(a);
  });
  if (mb_poison != adaptor.getInputs().end()) {
    return *mb_poison;
  } else if (llvm::all_of(adaptor.getInputs(), [](auto a) {
               return llvm::isa_and_present<mlir::TypedAttr>(a);
             })) {
    mlir::SmallVector<mlir::TypedAttr> as;
    llvm::transform(adaptor.getInputs(), std::back_inserter(as), [](auto a) {
      return llvm::dyn_cast<mlir::TypedAttr>(a);
    });
    return hugr_mlir::TupleAttr::get(getContext(), as);
  }
  return nullptr;
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

mlir::LogicalResult hugr_mlir::UnpackTupleOp::fold(
    FoldAdaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
  if (auto poison = llvm::dyn_cast_if_present<mlir::ub::PoisonAttrInterface>(
          adaptor.getInput())) {
    for (auto i = 0; i < getNumResults(); ++i) {
      results.push_back(poison);
    }
    return mlir::success();
  } else if (
      auto v =
          llvm::dyn_cast_if_present<hugr_mlir::TupleAttr>(adaptor.getInput())) {
    if (v.getValues().size() == getNumResults()) {
      llvm::copy(v.getValues(), std::back_inserter(results));
      return mlir::success();
    }
  }
  return mlir::failure();
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
    mlir::SmallVector<mlir::Type> expected_region_types{getInputs().getTypes()};

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
// TypeAliasOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::TypeAliasOp::verify() {
  // TODO
  // If aliasee: Verify aliasee HugrTypeInterface
  // If aliasee: Verify aliasee constraints and extensions match op
  if (auto aliasee = getAliaseeAttr()) {
    auto t = llvm::dyn_cast<HugrTypeInterface>(aliasee.getValue());
    if (!t) {
      return mlir::emitError(getLoc())
             << "aliasee is not HugrTypeInterface: " << aliasee.getValue();
    }
    if (t.getExtensions() != getExtensions()) {
      return mlir::emitError(getLoc())
             << "Extensions to not match aliasee: expected "
             << t.getExtensions() << " found " << getExtensions();
    }
    if (t.getConstraint() != getConstraint()) {
      return mlir::emitError(getLoc())
             << "Constraint to not match aliasee: expected "
             << t.getConstraintAttr() << " found " << getConstraintAttr();
    }
  }

  return mlir::success();
}

/////////////////////////////////////////////////////////////////////////////
// TagOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::TagOp::verify() {
  auto st = getOutput().getType();
  auto t = getTag().getSExtValue();
  if (t < 0 || t >= st.numAlts()) {
    return emitOpError("Tag out of bounds: ") << t;
  }
  if (getInput().getType() != st.getTypes()[t]) {
    return emitOpError("Input type doesn't match output type");
  }
  return mlir::success();
}

mlir::OpFoldResult hugr_mlir::TagOp::fold(FoldAdaptor adaptor) {
  if (auto poison = llvm::dyn_cast_if_present<mlir::ub::PoisonAttrInterface>(
          adaptor.getInput())) {
    return poison;
  } else if (
      auto v = llvm::dyn_cast_if_present<mlir::TypedAttr>(adaptor.getInput())) {
    return hugr_mlir::SumAttr::get(
        getResult().getType(), adaptor.getTag().getZExtValue(), v);
  }
  return nullptr;
}

/////////////////////////////////////////////////////////////////////////////
// ReadVariantOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::ReadVariantOp::inferReturnTypes(
    mlir::MLIRContext* context, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr attrs,
    mlir::OpaqueProperties props, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& result_types) {
  ReadVariantOp::Adaptor adaptor(operands, attrs, props, regions);
  auto st = llvm::dyn_cast<SumType>(adaptor.getInput().getType());
  if (!st) {
    return mlir::failure();
  }
  auto t = adaptor.getTag().getSExtValue();
  if (t < 0 || t >= st.numAlts()) {
    return mlir::failure();
  }
  result_types.push_back(st.getAltType(t));
  return mlir::success();
}

mlir::OpFoldResult hugr_mlir::ReadVariantOp::fold(FoldAdaptor adaptor) {
  if (auto poison = llvm::dyn_cast_if_present<mlir::ub::PoisonAttrInterface>(
          adaptor.getInput())) {
    return poison;
  } else if (
      auto attr =
          llvm::dyn_cast_if_present<hugr_mlir::SumAttr>(adaptor.getInput())) {
    if (getTag() == attr.getTag()) {
      return attr.getValue();
    }
    return mlir::ub::PoisonAttr::get(getContext());
  }
  return nullptr;
}

/////////////////////////////////////////////////////////////////////////////
// ReadTagOp
/////////////////////////////////////////////////////////////////////////////
mlir::OpFoldResult hugr_mlir::ReadTagOp::fold(FoldAdaptor adaptor) {
  if (auto poison = llvm::dyn_cast_if_present<mlir::ub::PoisonAttrInterface>(
          adaptor.getInput())) {
    return poison;
  } else if (
      auto attr =
          llvm::dyn_cast_if_present<hugr_mlir::SumAttr>(adaptor.getInput())) {
    return attr.getTagAttr();
  } else if (getInput().getType().numAlts() == 1) {
    return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), 0);
  }

  return nullptr;
}

/////////////////////////////////////////////////////////////////////////////
// ConstantOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::ConstantOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes, properties, regions);
  inferredReturnTypes.push_back(adaptor.getValue().getType());
  return mlir::success();
}

mlir::OpFoldResult hugr_mlir::ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValue();
}

/////////////////////////////////////////////////////////////////////////////
// UnpackFunctionOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::UnpackFunctionOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  mlir::OpBuilder builder(context);
  UnpackFunctionOpAdaptor adaptor(operands, attributes, properties, regions);
  auto old_ft = llvm::dyn_cast<FunctionType>(adaptor.getInput().getType());
  if (!old_ft) {
    return mlir::emitOptionalError(location, "Input not a function type:");
  }
  mlir::SmallVector<mlir::Type> new_arg_tys{builder.getType<ClosureType>()};
  llvm::copy(old_ft.getArgumentTypes(), std::back_inserter(new_arg_tys));
  inferredReturnTypes.push_back(
      builder.getFunctionType(new_arg_tys, old_ft.getResultTypes()));
  inferredReturnTypes.push_back(new_arg_tys[0]);
  return mlir::success();
}

/////////////////////////////////////////////////////////////////////////////
// UnpackFunctionOp
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::AllocFunctionOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  mlir::OpBuilder builder(context);
  AllocFunctionOpAdaptor adaptor(operands, attributes, properties, regions);
  auto func = adaptor.getFunc();
  FunctionType ft;
  if (!func || !(ft = llvm::dyn_cast<FunctionType>(func.getType()))) {
    return mlir::emitOptionalError(location, "func is not a static edge attr");
  }
  inferredReturnTypes.push_back(ft);
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
            mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>(
                e, v, LinearityResource::get()));
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

/////////////////////////////////////////////////////////////////////////////
//  verifyHugrSymbolUserOpInterface
/////////////////////////////////////////////////////////////////////////////
mlir::LogicalResult hugr_mlir::verifyHugrSymbolUses(
    mlir::Operation* op, HugrSymbolMap const& stc) {
  if (auto co = llvm::dyn_cast<CallOp>(op)) {
    if (mlir::failed(co.verifyHugrSymbolUses(stc))) {
      return mlir::failure();
    }
  }

  mlir::OpBuilder builder(op->getContext());

  llvm::SmallVector<std::tuple<mlir::Twine, mlir::Attribute>> worklist;
  llvm::transform(op->getAttrs(), std::back_inserter(worklist), [&](auto x) {
    return std::make_tuple(
        mlir::Twine("Attribute: ").concat(x.getName().getValue()),
        x.getValue());
  });
  for (auto& r : op->getRegions()) {
    for (auto const& [i, b] : llvm::enumerate(r.getBlocks())) {
      llvm::transform(
          b.getArguments(), std::back_inserter(worklist), [&](auto x) {
            mlir::Twine s = mlir::Twine("region,block,arg:")
                                .concat(mlir::Twine(r.getRegionNumber()))
                                .concat(",")
                                .concat(mlir::Twine(i))
                                .concat(",")
                                .concat(mlir::Twine(x.getArgNumber()));
            return std::make_tuple(s, mlir::TypeAttr::get(x.getType()));
          });
    }
  }
  llvm::transform(
      llvm::enumerate(op->getOperands()), std::back_inserter(worklist),
      [&](auto x) {
        auto [i, operand] = x;
        mlir::Twine s = mlir::Twine("operand ").concat(mlir::Twine(i));
        return std::make_tuple(s, mlir::TypeAttr::get(operand.getType()));
      });

  llvm::transform(
      llvm::enumerate(op->getResults()), std::back_inserter(worklist),
      [&](auto x) {
        auto [i, res] = x;
        mlir::Twine s = mlir::Twine("result ").concat(mlir::Twine(i));
        return std::make_tuple(s, mlir::TypeAttr::get(res.getType()));
      });

  for (auto const& [label, a] : worklist) {
    llvm::DenseSet<hugr_mlir::AliasRefType> ref_types;

    mlir::AttrTypeWalker walker;
    walker.addWalk([&](hugr_mlir::AliasRefType ref) { ref_types.insert(ref); });
    walker.walk(a);

    if (ref_types.empty()) {
      continue;
    }

    for (auto ref : ref_types) {
      std::optional<mlir::InFlightDiagnostic> mb_ifd;
      auto get_ifd = [&]() -> mlir::InFlightDiagnostic& {
        if (!mb_ifd) {
          mb_ifd.emplace(mlir::emitError(op->getLoc()))
              << "Error resolving type alias reference in " << label << " of "
              << op->getName() << ": " << ref;
        }
        return *mb_ifd;
      };
      auto referee = stc.lookup(ref.getRef());
      if (!referee) {
        return get_ifd() << "Unknown symbol: " << ref.getRef();
      }
      auto alias_op = llvm::dyn_cast<hugr_mlir::TypeAliasOp>(referee);
      if (!alias_op) {
        auto& ifd = get_ifd() << "Symbol references non type-alias op: "
                              << referee->getName();
        ifd.attachNote(referee->getLoc());
        return ifd;
      }

      if (alias_op.getExtensionsAttr() != ref.getExtensions()) {
        auto& ifd = get_ifd() << "Alias has mismatched extensions:"
                              << alias_op.getExtensionsAttr();
        ifd.attachNote(alias_op.getLoc());
        return ifd;
      }
      if (alias_op.getConstraint() != ref.getConstraint()) {
        auto& ifd = get_ifd() << "Alias has mismatched constraint:"
                              << alias_op.getConstraintAttr();
        ifd.attachNote(alias_op.getLoc());
        return ifd;
      }
    }
  }

  return mlir::success();
}

void hugr_mlir::HugrDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "hugr-mlir/IR/HugrOps.cpp.inc"
      >();
}
