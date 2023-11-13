#ifndef HUGR_MLIR_IR_TYPE_INTERFACES_H
#define HUGR_MLIR_IR_TYPE_INTERFACES_H

#include "hugr-mlir/IR/HugrAttrs.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

namespace hugr_mlir {
class ExtensionSetAttr;


}

#include "hugr-mlir/IR/HugrTypeInterfaces.h.inc"

namespace hugr_mlir {

struct LinearValue : mlir::TypedValue<HugrTypeInterface> {
    using TypedValue::TypedValue;
    static bool classof(mlir::Value v) {
        if(auto t = llvm::dyn_cast<HugrTypeInterface>(v.getType())) {
            return t.getConstraint() == TypeConstraint::Linear;
        }
        return false;
    }
};

template<typename RangeT>
auto linear_values_range(RangeT&& range) {
    return llvm::make_filter_range(llvm::map_range(range, [](auto x) { return llvm::dyn_cast<LinearValue>(x); }), [](auto x) { return !!x;});
}

}

#endif
