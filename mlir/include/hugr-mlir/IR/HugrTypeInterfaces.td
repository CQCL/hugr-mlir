#ifndef HUGR_MLIR_IR_HUGR_TYPE_INTERFACES_TD
#define HUGR_MLIR_IR_HUGR_TYPE_INTERFACES_TD

include "hugr-mlir/IR/HugrDialect.td"

def Hugr_TestTypeInterface : TypeInterface<"TestTypeInterface"> {
    let cppNamespace = "::hugr_mlir";
    let methods = [
        InterfaceMethod<[{
                Test interface
            }], "bool", "testInterface", (ins)>
    ];
}

#endif
