#include "hugr-mlir-c/Translate.h"
#include "hugr-mlir/Translate/HugrTranslate.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/SourceMgr.h"


int mlirHugrTranslateMain(int argc, char const* const* argv) {
   return mlir::failed(hugr_mlir::translateMain(argc, argv));
}

void mlirHugrRegisterTranslationToMLIR(MlirStringRef name, MlirStringRef description, mlirHugrTranslateStringRefToMLIRFunction translate_fun, mlirHugrDialectRegistrationFunction dialect_registration_fun) {
    std::string name_str = unwrap(name).str();
    std::string description_str = unwrap(description).str();
    // llvm::errs() << "mlirHugrRegisterTranslationToMLIR:" << unwrap(name) << "\n";
    mlir::TranslateToMLIRRegistration(
        name_str,
        description_str,
        [=](llvm::SourceMgr const& src, mlir::MLIRContext* context) -> mlir::OwningOpRef<mlir::Operation*> {
            context->loadAllAvailableDialects();
            // llvm::errs() << "mlirHugrRegisterTranslationToMLIR::translate" << name_str << "\n";
            auto& info = src.getBufferInfo(src.getMainFileID());
            mlir::OpBuilder builder(context);
            mlir::Location loc = mlir::FileLineColLoc::get(builder.getStringAttr(info.Buffer->getBufferIdentifier()), 0, 0);
            return mlir::OwningOpRef(unwrap(translate_fun(wrap(info.Buffer->getBuffer()), wrap(loc))));
        }, [=](mlir::DialectRegistry& registry) {
            // llvm::errs() << "mlirHugrRegisterTranslationToMLIR::register" << name_str << "\n";
            dialect_registration_fun(wrap(&registry));
        }
   );
}
