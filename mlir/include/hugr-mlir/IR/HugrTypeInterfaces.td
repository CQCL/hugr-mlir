#ifndef HUGR_MLIR_IR_HUGR_TYPE_INTERFACES_TD
#define HUGR_MLIR_IR_HUGR_TYPE_INTERFACES_TD

include "hugr-mlir/IR/HugrDialect.td"

def HugrTypeInterface : TypeInterface<"HugrTypeInterface"> {
    let cppNamespace = Hugr_Dialect.cppNamespace;
    let description = [{
        All `hugr` ops take arguments implementing `HugrTypeInterface`. This
        allows them to inspect the hugr extensions and hugr type constraints of
        their arguments.

        TODO: It may be necessary to support types that can sometimes report
        their extensions and constraints. For example `tuple` can satisfy this
        interface if and only if it's members can.
    }];
    let methods = [
        InterfaceMethod<[{
                TODO Description,
            }], "hugr_mlir::ExtensionSetAttr", "getExtensions", (ins)>,
        InterfaceMethod<[{
                TODO Description,
            }], "hugr_mlir::TypeConstraint", "getConstraint", (ins)>
    ];
}

#endif
