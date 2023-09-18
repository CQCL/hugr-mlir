// RUN: test-hugr-rs-bridge | FileCheck %s
#include "hugr-rs-bridge/hugr-rs-bridge.h"

int main(int argc, char* argv[]) {
    llvm::outs() << "TestHugrRsBridge\n";
    using namespace hugr_rs_bridge;

    mlir::MLIRContext context;
    auto unknown_loc = mlir::UnknownLoc::get(&context);

    auto h0 = get_example_hugr();
    llvm::outs() << "We have an example hugr\n";

    auto hugr_rmp = hugr_to_rmp(unknown_loc, *h0);
    if(mlir::failed(hugr_rmp)) { return 1;  }
    llvm::outs() << "We have serialized the hugr to msgpack\n";
    auto h1 = parse_hugr_rmp(unknown_loc, *hugr_rmp);
    if(mlir::failed(h1)) { return 1;  }
    llvm::outs() << "We have parsed the msgpack\n";

    auto hugr_json = hugr_to_json(unknown_loc, **h1);
    if(mlir::failed(h1)) { return 1;  }
    llvm::outs() << "We have serialized the hugr to json\n";
    llvm::outs() << hugr_json;
}

// CHECK-LABEL: TestHugrRsBridge
// CHECK: We have an example hugr
// CHECK: We have serialized the hugr to msgpack
// CHECK: We have parsed the msgpack
// CHECK: We have serialized the hugr to json
