// RUN: test-import-example-hugr 2>&1 | FileCheck %s

#include "hugr-rs-bridge/hugr-rs-bridge.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"

int main() {
    using namespace mlir;
    DialectRegistry registry;
    registry.insert<hugr_mlir::HugrDialect>();
    MLIRContext context(registry);
    context.getOrLoadDialect("hugr");

    auto hugr = hugr_rs_bridge::get_example_hugr();

    OpBuilder builder(&context);
    auto mod = builder.create<hugr_mlir::ModuleOp>(builder.getUnknownLoc());
    if(failed(hugr_rs_bridge::hugr_to_mlir(&context, *hugr, &mod.getBody().front()))) { return 1; }

    mod->dump();
    return 0;

}

// CHECK: hello
