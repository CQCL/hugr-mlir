#include "hugr-mlir-c/Translate.h"

#include "hugr-mlir/Translate/HugrTranslate.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-translate/Translation.h"

int mlirHugrTranslateMain(int argc, char const* const* argv) {
  return mlir::failed(hugr_mlir::translateMain(argc, argv));
}

void mlirHugrRegisterTranslationToMLIR(
    MlirStringRef name, MlirStringRef description,
    mlirHugrTranslateStringRefToMLIRFunction translate_fun,
    mlirHugrDialectRegistrationFunction dialect_registration_fun) {
  std::string name_str = unwrap(name).str();
  std::string description_str = unwrap(description).str();
  // llvm::errs() << "mlirHugrRegisterTranslationToMLIR:" << unwrap(name) <<
  // "\n";
  mlir::TranslateToMLIRRegistration(
      name_str, description_str,
      [=](llvm::SourceMgr const& src,
          mlir::MLIRContext* context) -> mlir::OwningOpRef<mlir::Operation*> {
        context->loadAllAvailableDialects();
        // llvm::errs() << "mlirHugrRegisterTranslationToMLIR::translate" <<
        // name_str << "\n";
        auto& info = src.getBufferInfo(src.getMainFileID());
        mlir::OpBuilder builder(context);
        mlir::Location loc = mlir::FileLineColLoc::get(
            builder.getStringAttr(info.Buffer->getBufferIdentifier()), 0, 0);
        return mlir::OwningOpRef(
            unwrap(translate_fun(wrap(info.Buffer->getBuffer()), wrap(loc))));
      },
      [=](mlir::DialectRegistry& registry) {
        // llvm::errs() << "mlirHugrRegisterTranslationToMLIR::register" <<
        // name_str << "\n";
        dialect_registration_fun(wrap(&registry));
      });
}

struct EmitContext {
  std::reference_wrapper<llvm::raw_ostream> ostream;
};

void mlirHugrEmitStringRef(struct EmitContext const* emit_context, MlirStringRef str) {
  emit_context->ostream.get() << unwrap(str);
}

void mlirHugrRegisterTranslationFromMLIR(
    MlirStringRef name, MlirStringRef description,
    mlirHugrTranslateFromMLIRFunction translate_fun,
    mlirHugrDialectRegistrationFunction dialect_registration_fun) {

  std::string name_str = unwrap(name).str();
  std::string description_str = unwrap(description).str();

  mlir::TranslateFromMLIRRegistration(
      name_str, description_str,
      [=](mlir::Operation* op, llvm::raw_ostream& os) -> mlir::LogicalResult {
        EmitContext ctx{os};
        return unwrap(translate_fun(wrap(op), &ctx));
      },[=](mlir::DialectRegistry& registry) {
        dialect_registration_fun(wrap(&registry));
      }
  );
}
