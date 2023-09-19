#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  return failed(
      mlir::mlirTranslateMain(argc, argv, "hugr-mlir translation tool"));
}
