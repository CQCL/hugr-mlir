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

  mlir_type_subclass(hugrM, "FunctionType", mlirTypeIsAHugrFunctionType)
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute extensions, MlirType function_type) {
            return cls(mlirHugrFunctionTypeGet(extensions, function_type));
          },
          "cls"_a, "extensions"_a, "function_type"_a);

  mlir_attribute_subclass(hugrM, "ExtensionAttr", mlirAttributeIsAHugrExtensionAttr)
    .def_classmethod("get", [](py::object cls, std::string const& name ,MlirContext context) {
      return cls(mlirHugrExtensionAttrGet(context, mlirStringRefCreateFromCString(name.c_str())));
    }, "cls"_a, "name"_a, "context"_a = py::none());

  mlir_attribute_subclass(hugrM, "ExtensionSetAttr", mlirAttributeIsAHugrExtensionSetAttr)
    .def_classmethod("get", [](py::object cls, std::vector<MlirAttribute> const& extensions, MlirContext context) {
      return cls(mlirHugrExtensionSetAttrGet(context, extensions.size(), extensions.data()));
    }, "cls"_a, "extensions"_a = py::list(), "context"_a = py::none());

}
