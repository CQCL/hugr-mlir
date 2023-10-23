#include "hugr-mlir/Translate/HugrTranslate.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

mlir::LogicalResult hugr_mlir::translateMain(
    int argc, char const* const* argv) {
  // llvm::errs() << "hugr_mlir::translateMain " << argc;
  // for(auto i = 0; i < argc; ++i) {
  //     llvm::errs() << "[" << argv[i] << "]";
  // }
  // llvm::errs() << "\n";
  return mlir::mlirTranslateMain(
      argc, const_cast<char**>(argv), "hugr-mlir translation tool");
}
