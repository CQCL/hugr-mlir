// RUN: hugr-mlir-opt %s --lower-hugr=hugr-verify --convert-hugr | FileCheck %s

module {
  hugr.module {
    func @test[](%arg0: !hugr.sum<tuple<>, tuple<>>) -> !hugr<opaque "qubit"["prelude"] [Linear]> {
      %0 = cfg %arg0 : (!hugr.sum<tuple<>, tuple<>>) -> !hugr<opaque "qubit"["prelude"] [Linear]> {
      ^bb0(%arg1: !hugr.sum<tuple<>, tuple<>>):
        switch %arg1 : !hugr.sum<tuple<>, tuple<>> ^bb2, ^bb3
      ^bb1(%1: !hugr<opaque "qubit"["prelude"] [Linear]>):  // pred: ^bb4
        output %1 : !hugr<opaque "qubit"["prelude"] [Linear]>
      ^bb2:  // pred: ^bb0
        %2 = call @new  []() -> !hugr<opaque "qubit"["prelude"] [Linear]>
        %3 = ext_op []"H" %2 : (!hugr<opaque "qubit"["prelude"] [Linear]>) -> !hugr<opaque "qubit"["prelude"] [Linear]>
        %4 = make_tuple()
        %5 = tag 0 %4 : tuple<> -> <tuple<>>
        switch %5, %3 : !hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]> ^bb4
      ^bb3:  // pred: ^bb0
        %6 = call @new  []() -> !hugr<opaque "qubit"["prelude"] [Linear]>
        %7 = make_tuple()
        %8 = tag 0 %7 : tuple<> -> <tuple<>>
        switch %8, %6 : !hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]> ^bb4
      ^bb4(%9: !hugr<opaque "qubit"["prelude"] [Linear]>):  // 2 preds: ^bb2, ^bb3
        %10 = make_tuple()
        %11 = tag 0 %10 : tuple<> -> <tuple<>>
        switch %11, %9 : !hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]> ^bb1
      }
      output %0 : !hugr<opaque "qubit"["prelude"] [Linear]>
    }
    func @new[]() -> !hugr<opaque "qubit"["prelude"] [Linear]>
  }
}

// CHECK-LABEL: module {
// CHECK-NOT: cfg
