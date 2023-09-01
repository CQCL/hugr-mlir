// RUN: hugr-mlir-opt %s | FileCheck %s

!unit = tuple<>
!bool = !hugr.sum<!unit,!unit>
hugr.module @basic_module_cfg {
  func @main [](%x: index) -> index {
     const @simple_unary_predicate : !bool
     %y = cfg %x : (index) -> index {
     ^bb0(%a : index): // The entry block may not have successors, so we introduce this dummy block and cf.br to the real entry block
       cf.br ^bb1(%a : index)
     ^bb1(%b : index):
       %i = make_tuple (%b : index)
       %j = tag 1 %i : tuple<index> -> !hugr.sum<tuple<index>,tuple<index>>
       switch %j : !hugr.sum<tuple<index>,tuple<index>> ^bb2,^bb3
     ^bb2(%c : index):
       %k = load_constant @simple_unary_predicate : !bool
       switch %k, %c : !bool, index ^bb1, ^bb3
     ^bb3(%d : index):
       output %d : index
     }
     output %y : index
  }
}

// CHECK-LABEL: hugr.module @basic_module_cfg
