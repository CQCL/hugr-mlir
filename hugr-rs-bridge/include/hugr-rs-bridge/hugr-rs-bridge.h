#ifndef HUGR_RS_BRIDGE_HUGR_RS_BRIDGE_H
#define HUGR_RS_BRIDGE_HUGR_RS_BRIDGE_H

// clang-format off
#include <memory>

#include "rust/cxx.h"
#include "hugr-rs-bridge/src/lib.rs.h"
// clang-format on

namespace mlir {
class Location;
}

namespace hugr_rs_bridge {

using Hugr = detail::Hugr;

template <typename T>
struct BoxDeleter {
  void operator()(T* t) { ::rust::Box<T>::from_raw(t); }
};

template <typename T>
using hugr_unique_ptr = std::unique_ptr<T, BoxDeleter<T>>;

mlir::FailureOr<hugr_unique_ptr<Hugr>> parse_hugr_json(
    mlir::Location loc, llvm::StringRef str);
mlir::FailureOr<hugr_unique_ptr<Hugr>> parse_hugr_rmp(
    mlir::Location loc, llvm::ArrayRef<uint8_t>);

mlir::FailureOr<std::string> hugr_to_json(mlir::Location loc, Hugr const&);
mlir::FailureOr<std::vector<uint8_t>> hugr_to_rmp(
    mlir::Location loc, Hugr const&);

hugr_unique_ptr<Hugr> get_example_hugr();

}  // namespace hugr_rs_bridge

#endif  // HUGR_RS_BRIDGE_HUGR_RS_BRIDGE_H
