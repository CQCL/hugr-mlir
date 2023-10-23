#include "hugr-mlir-c/Dialects.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_hugrDialects, m) {
  m.doc() = "hugr_mlir Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  auto hugrM = m.def_submodule("hugr");

  hugrM.def(
      "register_dialect",
      [](py::object capsule, bool load) {
        auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
        MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());
        assert(!mlirContextIsNull(context) && "context must not be null");

        MlirDialectHandle handle = mlirGetDialectHandle__hugr__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
