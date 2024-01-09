#ifndef HUGR_MLIR_OPT_HUGROPT_H
#define HUGR_MLIR_OPT_HUGROPT_H

namespace mlir {
class DialectRegistry;
}

namespace hugr_mlir {

int HugrMlirOptMain(int argc, char **argv, mlir::DialectRegistry *);
void registerHugrOptPipelines();

}  // namespace hugr_mlir

#endif  // HUGR_MLIR_OPT_HUGROPT_H
