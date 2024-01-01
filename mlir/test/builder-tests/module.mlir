// RUN: hugr-mlir-opt %s | FileCheck %s

hugr.module @basic_recurse {
  func @main[](%x: index) -> index {
    %y = call @main[](index) -> index %x
    output %y: index
  }
}

// CHECK-LABEL: hugr.module @basic_recurse
// CHECK: func @main[](%{{.*}}: index) -> index

!qubit_alias = !hugr.ref<["quantum"]@qubit_state,Linear>
hugr.module @simple_alias {
  type_alias ["quantum"]@qubit_state, Linear
  func @main[](!qubit_alias) -> !qubit_alias
}

// CHECK-LABEL: hugr.module @simple_alias
// CHECK: type_alias ["quantum"]@qubit_state, Linear
// CHECK: func @main[](!hugr.ref<["quantum"]@qubit_state, Linear>) -> !hugr.ref<["quantum"]@qubit_state, Linear>

hugr.module @local_def {
  func @main[](%x: index) -> index {
    %y = dfg %x : (index) -> index {
    ^bb0(%x1: index):
      func @local[](index) -> index
      %y = call @local [](index) -> index %x1
      output %y : index
    }
    output %y : index
  }
}

// CHECK-LABEL: hugr.module @local_def
// CHECK: func @main[]({{.*}}: index) -> index {
// CHECK: func @local[](index) -> index
// CHECK: call @local [](index) -> index
