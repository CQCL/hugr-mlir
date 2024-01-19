#ifndef HUGR_MLIR_CONVERSION_UTILS_H
#define HUGR_MLIR_CONVERSION_UTILS_H

#include "hugr-mlir/IR/HugrOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hugr_mlir {

inline bool isTopLevelFunc(hugr_mlir::FuncOp op) {
  return op && llvm::isa_and_present<hugr_mlir::ModuleOp>(op->getParentOp());
}

struct FuncClosureMap {
  hugr_mlir::FuncOp lookupTopLevelFunc(mlir::SymbolRefAttr ref) const {
    if(!ref.getNestedReferences().empty()) { return nullptr; }
    return lookupTopLevelFunc(ref.getLeafReference().getValue());
  }
  hugr_mlir::FuncOp lookupTopLevelFunc(mlir::StringRef ref) const {
    if(!func_map.contains(ref)) { return nullptr; }
    if(auto x = func_map.lookup(ref)) {
      return x;
    }
    return nullptr;
  }

  hugr_mlir::AllocFunctionOp lookupAllocFunctionOp(mlir::SymbolRefAttr ref) const {
    if(!ref.getNestedReferences().empty()) { return nullptr; }
    return lookupAllocFunctionOp(ref.getLeafReference().getValue());
  }
  hugr_mlir::AllocFunctionOp lookupAllocFunctionOp(mlir::StringRef ref) const {
    if(!alloc_map.contains(ref)) { return nullptr; }
    if(auto x = alloc_map.lookup(ref)) {
      return x;
    }
    return nullptr;
  }

  mlir::LogicalResult insert(hugr_mlir::FuncOp func_op) {
    assert(isTopLevelFunc(func_op) && "Precondition");
    return mlir::success(func_map.try_emplace(func_op.getSymName(), func_op).second);
  }

  mlir::LogicalResult insert(hugr_mlir::AllocFunctionOp alloc_op) {
      auto sym = alloc_op.getFunc().getRef();
      if(sym.getNestedReferences().size()) { return mlir::failure(); }
      return mlir::success(alloc_map.try_emplace(sym.getLeafReference().getValue(), alloc_op).second);
  }

private:
  using AllocMap_t = llvm::DenseMap<mlir::StringRef, hugr_mlir::AllocFunctionOp>;
  using TopLevelMap_t = llvm::DenseMap<mlir::StringRef, hugr_mlir::FuncOp>;
  AllocMap_t alloc_map;
  TopLevelMap_t func_map;
};

template<typename OpT>
struct FuncClosureMapOpConversionPatternBase : mlir::OpConversionPattern<OpT> {
  template <typename... Args>
  FuncClosureMapOpConversionPatternBase(
      hugr_mlir::FuncClosureMap const& fcm, Args&&... args)
      : mlir::OpConversionPattern<OpT>(std::forward<Args>(args)...),
        func_closure_map(fcm) {}

protected:
  hugr_mlir::FuncOp lookupTopLevelFunc(mlir::SymbolRefAttr ref) const {
    if(!ref.getNestedReferences().empty()) { return nullptr; }
    return func_closure_map.lookupTopLevelFunc(ref.getLeafReference());
  }

  hugr_mlir::AllocFunctionOp lookupAllocFunctionOp(mlir::SymbolRefAttr ref) const {
    if(!ref.getNestedReferences().empty()) { return nullptr; }
    return func_closure_map.lookupAllocFunctionOp(ref.getLeafReference());
  }

private:
  hugr_mlir::FuncClosureMap const& func_closure_map;
};

struct ExtensionOpMatcher {
  ExtensionOpMatcher(mlir::StringRef extname_, mlir::StringRef opname_, int num_args_, int num_results_)
      : opname(opname_), extname(extname_), num_args(num_args_), num_results(num_results_) {}

  mlir::LogicalResult match(mlir::RewriterBase&, hugr_mlir::ExtensionOp) const;
  mlir::StringRef const opname;
  mlir::StringRef const extname;
  int const num_args;
  int const num_results;
};

struct HugrExtensionOpRewritePattern : mlir::OpRewritePattern<hugr_mlir::ExtensionOp> {
  template<typename ...Args>
  HugrExtensionOpRewritePattern(mlir::StringRef extname, mlir::StringRef opname, int num_args, int num_results, Args&& ...args) :
    matcher(extname, opname, num_args, num_results), OpRewritePattern(std::forward<Args>(args)...) {}

  mlir::LogicalResult matchAndRewrite(hugr_mlir::ExtensionOp, mlir::PatternRewriter&) const override;
  virtual void replace(hugr_mlir::ExtensionOp, mlir::PatternRewriter&) const = 0;
protected:
  ExtensionOpMatcher matcher;
};

struct HugrExtensionOpConversionPattern : mlir::OpConversionPattern<hugr_mlir::ExtensionOp> {
  template<typename ...Args>
  HugrExtensionOpConversionPattern(mlir::StringRef extname, mlir::StringRef opname, int num_args, int num_results, Args&& ...args) :
    matcher(extname, opname, num_args, num_results), OpConversionPattern(std::forward<Args>(args)...) {}

  mlir::LogicalResult matchAndRewrite(hugr_mlir::ExtensionOp, OpAdaptor, mlir::ConversionPatternRewriter&) const override;
  virtual mlir::LogicalResult replace(hugr_mlir::ExtensionOp, OpAdaptor, mlir::ConversionPatternRewriter&) const = 0;
protected:
  ExtensionOpMatcher matcher;
};


}



#endif // HUGR_MLIR_CONVERSION_UTILS_H
