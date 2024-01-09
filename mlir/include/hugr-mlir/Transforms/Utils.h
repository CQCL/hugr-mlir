#ifndef HUGR_MLIR_TRANSFORMS_UTILS_H
#define HUGR_MLIR_TRANSFORMS_UTILS_H

namespace mlir {
class Operation;
}

namespace hugr_mlir {
bool opHasNonLocalEdges(mlir::Operation*);
}

#endif  // UTILS_H_
