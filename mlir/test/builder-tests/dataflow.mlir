// RUN: hugr-mlir-opt %s | FileCheck %s

!unit = tuple<>
!bool = !hugr.sum<!unit,!unit>

hugr.module @nested_identity {
  type_alias ["quantum"]@qb, Linear
  func @main[](%i: index, %qb0: !hugr.ref<["quantum"]@qb,Linear>) -> (index, !hugr.ref<["quantum"]@qb,Linear>) {
    %qb1 = ext_op ["quantum"] "H"  %qb0 : (!hugr.ref<["quantum"]@qb,Linear>) -> (!hugr.ref<["quantum"]@qb,Linear>)

    %j = dfg %i : (index) -> index  {
    ^bb0(%a: index):
       output %a : index
    }

    output %j, %qb1 : index, !hugr.ref<["quantum"]@qb,Linear>
  }
}

// CHECK-LABEL: hugr.module @nested_identity

hugr.module @copy_insertion_1 {
  func @main [](%x : !bool) -> (!bool,!bool) {
    output %x,%x : !bool,!bool
  }
}

// CHECK-LABEL: hugr.module @copy_insertion_1

hugr.module @copy_insertion_2 {
  func @main [](%x : !bool) -> (!bool,!bool) {
    %xor = ext_op ["arith"] "and" %x, %x : (!bool,!bool) -> !bool // this is not an xor :/
    output %xor, %x : !bool, !bool
  }
}


// CHECK-LABEL: hugr.module @copy_insertion_2
//
hugr.module @copy_insertion_3 {
  func @main [](%x : !bool) -> (!bool,!bool) {
    %xor1 = ext_op ["arith"] "and" %x, %x : (!bool,!bool) -> !bool // this is not an xor :/
    %xor2 = ext_op ["arith"] "and" %x, %xor1 : (!bool,!bool) -> !bool // this is not an xor :/
    output %xor2, %x : !bool, !bool
  }
}

// CHECK-LABEL: hugr.module @copy_insertion_3

hugr.module @simple_inter_graph_edge {
  func @main [](%x : index) -> index {
    %i1 = identity %x : index
    %nested = dfg : () -> index {
      %id = identity %i1 : index
      output %id : index
    }
    output %nested : index
  }
}

// CHECK-LABEL: hugr.module @simple_inter_graph_edge

// TODO insert_hugr: this just tests metadata

hugr.module @lift_node {
  func @main ["A","B","C"](%in : index) -> index {
    %out = dfg %in : (index) -> !hugr.ext<["A","B","C"]index> {
    ^bb0(%i : index):
      %add_ab = dfg %i : (index) -> !hugr.ext<["A","B"]index> {
      ^bb1(%j : index):
        %lift_a = lift ["A"] %j :  (index) -> !hugr.ext<["A"]index>
        %lift_b = lift ["B"] %lift_a : (!hugr.ext<["A"]index>) -> !hugr.ext<["A","B"]index>
        output %lift_b : !hugr.ext<["A","B"]index>
      }
      %add_c = dfg input extensions ["A","B"] %add_ab : (!hugr.ext<["A","B"]index>)-> !hugr.ext<["A","B","C"]index> {
      ^bb2(%k : index):
        %lift_c = lift ["C"] %k : (index) -> !hugr.ext<["C"]index>
        output %lift_c : !hugr.ext<["C"]index>
      }
      output %add_c : !hugr.ext<["A","B","C"]index>
    }
    output %out : !hugr.ext<["A","B","C"]index>
  }
}

// CHECK-LABEL: hugr.module @lift_node
