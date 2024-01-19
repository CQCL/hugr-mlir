#include "hugr-mlir/Conversion/Utils.h"

using namespace mlir;


LogicalResult hugr_mlir::ExtensionOpMatcher::match(mlir::RewriterBase &rw, hugr_mlir::ExtensionOp op) const {
    if(!op.getExtension().getName().equals(extname) || !op.getHugrOpname().equals(opname) || op.getNumOperands() != num_args || op.getNumResults() != num_results) {
        return rw.notifyMatchFailure(op, [&](auto& d) {
            d << "Expected (" << opname << "," << extname << "," << num_args << "," << num_results << ")\n";
            d << "Found (" << op.getHugrOpname() << "," << op.getExtension().getName() << "," << op.getNumOperands() << "," << op.getNumResults() << ")\n";
        });
    }
    return success();
}

LogicalResult hugr_mlir::HugrExtensionOpRewritePattern::matchAndRewrite(hugr_mlir::ExtensionOp op, mlir::PatternRewriter& rw) const {
    auto lr = matcher.match(rw, op);
    if(failed(lr)) { return lr; }
    replace(op, rw);
    return mlir::success();
}

LogicalResult hugr_mlir::HugrExtensionOpConversionPattern::matchAndRewrite(hugr_mlir::ExtensionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rw) const {
    auto lr = matcher.match(rw, op);
    if(failed(lr)) { return lr; }
    return replace(op, adaptor, rw);
}
