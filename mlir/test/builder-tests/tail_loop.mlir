// RUN: hugr-mlir-opt %s --verify-each | FileCheck %s

!unit = tuple<>
!bool = !hugr.sum<!unit,!unit>
hugr.module @basic_loop {
    func @foo [](%x : !bool) -> (index, !bool) {
        %i, %j = tailloop !hugr.sum<!unit,tuple<index>> passthrough (%x : !bool) -> (index) {
        ^bb0():
        %a = arith.constant 1 : index
        %b = tag 1 %a : index -> <!unit,tuple<index>>
        output %b : !hugr.sum<!unit,tuple<index>>
        }
        output %i, %j : index, !bool
    }
}

// CHECK-LABEL: hugr.module @basic_loop
// func @foo
// CHECK: = tailloop !hugr.sum<tuple<>, tuple<index>> passthrough({{.*}}) -> (index) {

!either_index_index = !hugr.sum<tuple<index>,tuple<index>>

hugr.module @loop_with_conditional {
    func @main[](%x: index) -> index {
         %r = tailloop !either_index_index (%x : index) -> (index) {
         ^bb0(%y : index):
           %unit = make_tuple ()
           %true = tag 1 %unit : !unit -> !bool
           %s = conditional (%true, %x : !bool, index) -> !either_index_index {
           ^bb1(%z: index):
             %a = make_tuple (%z : index)
             %b = tag 0 %a : tuple<index> -> !either_index_index
             output %b : !hugr.sum<tuple<index>,tuple<index>>
           }, {
           ^bb2(%_ : index):
             %two = arith.constant 2 : index
             %c = make_tuple (%two : index)
             %d = tag 1 %c : tuple<index> -> !either_index_index
             output %d : !either_index_index
           }
           output %s : !either_index_index
        }
        output %r : index
    }
}

// CHECK-LABEL: hugr.module @loop_with_conditional
