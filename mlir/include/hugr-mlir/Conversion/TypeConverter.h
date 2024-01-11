#ifndef HUGR_MLIR_CONVERSION_TYPECONVERTER_H
#define HUGR_MLIR_CONVERSION_TYPECONVERTER_H

#include <memory>

namespace mlir {
class RewritePatternSet;
class ConversionTarget;
class TypeConverter;
class MLIRContext;
class OneToNTypeConverter;
}  // namespace mlir

namespace hugr_mlir {
/* void populateHugrToLLVMConversionPatterns(mlir::RewritePatternSet&,
 * mlir::TypeConverter const&, int benefit = 1); */
std::unique_ptr<mlir::OneToNTypeConverter> createTypeConverter();
std::unique_ptr<mlir::TypeConverter> createSimpleTypeConverter();
}  // namespace hugr_mlir

#endif
