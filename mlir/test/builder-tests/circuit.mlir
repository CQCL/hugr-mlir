// RUN: hugr-mlir-opt %s | FileCheck %s

// TODO once we have an analysis that verifies linearity run it on this file

!qubit = !hugr.ref<["quantum"]@qubit_state,Linear>

hugr.module @simple_linear {
  type_alias ["quantum"]@qubit_state, Linear
  func @main["quantum"](%x0: !qubit, %y0: !qubit) -> (!qubit,!qubit) {
    %x1 = ext_op ["quantum"] "H" %x0 : (!qubit) -> !qubit
    %x2, %y2 = ext_op ["quantum"] "CX" %x1, %y0 : (!qubit,!qubit) -> (!qubit,!qubit)
    %x3, %y3 = ext_op ["quantum"] "CX" %y2, %x2 : (!qubit,!qubit) -> (!qubit,!qubit)
    output %x3, %y3: !qubit, !qubit
  }
}

// CHECK-LABEL: hugr.module @simple_linear

!unit = tuple<>
!bool = !hugr.sum<!unit,!unit>

hugr.module @with_nonlinear_and_outputs {
  type_alias ["quantum"]@qubit_state, Linear
  func @main["quantum","MissingRsrc"](%x0: !qubit, %y0: !qubit, %angle: index) -> (!qubit, !qubit, !bool) {
    %x1, %y1 = ext_op ["quantum"] "CX" %x0, %y0 : (!qubit, !qubit) -> (!qubit, !qubit)
    %x2, %measurement = ext_op ["MissingRsrc"] "MyOp" %x1, %angle: (!qubit, index) -> (!qubit, !bool)
    output %x2, %y1, %measurement: !qubit,!qubit,!bool
  }
}

// CHECK-LABEL: hugr.module @with_nonlinear_and_outputs
