#ifndef HUGR_MLIR_TARGET_HUGR_IMPORT_H
#define HUGR_MLIR_TARGET_HUGR_IMPORT_H

#include "mlir/IR/OwningOpRef.h"
namespace mlir {
class MLIRContext;
}
namespace llvm {
class StringRef;
}

namespace hugr_mlir {
/// Translates the LLVM module into an MLIR module living in the given context.
/// The translation supports operations from any dialect that has a registered
/// implementation of the LLVMImportDialectInterface. It returns nullptr if the
/// translation fails and reports errors using the error handler registered with
/// the MLIR context. The `emitExpensiveWarnings` option controls if expensive
/// but uncritical diagnostics should be emitted.
mlir::OwningOpRef<mlir::Operation *> translateHugrToMLIR(
    llvm::StringRef, mlir::MLIRContext *context);

void registerHugrToMLIRTranslations();

}  // namespace hugr_mlir

#endif
