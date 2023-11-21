// RUN: hugr-mlir-opt %s | FileCheck %s

!unit = tuple<>
!bool = !hugr.sum<!unit,!unit>

hugr.module @basic_conditional {
    hugr.func @foo [](%p : !bool) -> index {
        %i = conditional(%p : !bool) -> index
        output %i : index
    }
}

// CHECK-LABEL: hugr.module @basic_conditional
// CHECK: func
// CHECK: = conditional({{.*}} : !hugr.sum<tuple<>, tuple<>>) -> index
// CHECK: output %

hugr.module @basic_conditional_module {
    func @main[](%x: index) -> index {
         const @c : !unit = #hugr.unit
         %y = load_constant @c : !bool
         %z = conditional (%y, %x : !bool, index) -> index {
         ^bb0(%a: index):
           output %a : index
         }, {
         ^bb0(%a: index):
           output %a : index
         }
         output %z : index
    }
}

// CHECK-LABEL: hugr.module @basic_conditional_module
