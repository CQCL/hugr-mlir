// #include "hugr-rs-bridge/hugr-rs-bridge.h"

#include "hugr-mlir/IR/HugrDialect.h"
#include "hugr-mlir/IR/HugrOps.h"

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/SourceMgr.h"


// static mlir::OwningOpRef<hugr_mlir::ModuleOp> translateHugrRmpToMlir(std::shared_ptr<llvm::SourceMgr> mgr, mlir::MLIRContext* context) {
//     using namespace mlir;
//     auto n = mgr->getNumBuffers();
//     OpBuilder builder(context);
//     OwningOpRef<hugr_mlir::ModuleOp> x(builder.create<hugr_mlir::ModuleOp>(UnknownLoc::get(context)));
//     auto block = &x->getBody().front();
//     for(auto i = 0u; i < n; ++i) {
//         auto& info = mgr->getBufferInfo(i);
//         auto loc = FileLineColLoc::get(context, info.Buffer->getBufferIdentifier(), 0, 0);
//         auto array = llvm::ArrayRef<uint8_t>(info.Buffer->getBuffer().bytes_begin(), info.Buffer->getBuffer().bytes_end());
//         auto hugr = hugr_rs_bridge::parse_hugr_rmp(loc, array);
//         if(failed(hugr)) { return {}; }

//         if(failed(hugr_rs_bridge::hugr_to_mlir(context, **hugr, block))) { return {}; }
//     }
//     return x;
// }
// namespace {
// using namespace mlir;
// void registerHugrToMlir() {
//   TranslateToMLIRRegistration hugr_to_mlir(
//       "hugr-to-mlir", "translate hugr to mlir",
//       [](std::shared_ptr<llvm::SourceMgr> mgr, MLIRContext* context) -> OwningOpRef<Operation*> {
//           return translateHugrRmpToMlir(mgr, context);
//       },
//       [](DialectRegistry &registry) {
//         registry.insert<hugr_mlir::HugrDialect>();
//       });
// }
// }

int main(int argc, char **argv) {
  // registerHugrToMlir();
  return failed(
      mlir::mlirTranslateMain(argc, argv, "hugr-mlir translation tool"));
}
