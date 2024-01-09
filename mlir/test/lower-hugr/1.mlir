// RUN: hugr-mlir-opt %s --lower-hugr | FileCheck %s

module {
  hugr.module {
    func @test[](%arg0: !hugr.sum<tuple<>, tuple<>>) -> !hugr<opaque "qubit"["prelude"] [Linear]> {
      %0 = dfg input extensions [] %arg0 : (!hugr.sum<tuple<>, tuple<>>) -> !hugr<opaque "qubit"["prelude"] [Linear]> {
      ^bb0(%arg1: !hugr.sum<tuple<>, tuple<>>):
        %1 = cfg %arg1 : (!hugr.sum<tuple<>, tuple<>>) -> !hugr<opaque "qubit"["prelude"] [Linear]> {
        ^bb0(%arg2: !hugr.sum<tuple<>, tuple<>>):
          %2 = dfg input extensions [] %arg2 : (!hugr.sum<tuple<>, tuple<>>) -> !hugr.sum<tuple<>, tuple<>> {
          ^bb0(%arg3: !hugr.sum<tuple<>, tuple<>>):
            output %arg3 : !hugr.sum<tuple<>, tuple<>>
          }
          switch %2 : !hugr.sum<tuple<>, tuple<>> ^bb2, ^bb3
        ^bb1(%3: !hugr<opaque "qubit"["prelude"] [Linear]>):  // pred: ^bb4
          output %3 : !hugr<opaque "qubit"["prelude"] [Linear]>
        ^bb2:  // pred: ^bb0
          %4:2 = dfg input extensions []  : () -> (!hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]>) {
            %8 = call @new  []() -> !hugr<opaque "qubit"["prelude"] [Linear]>
            %9 = ext_op []"H" %8 : (!hugr<opaque "qubit"["prelude"] [Linear]>) -> !hugr<opaque "qubit"["prelude"] [Linear]>
            %10 = make_tuple()
            %11 = tag 0 %10 : tuple<> -> <tuple<>>
            output %11, %9 : !hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]>
          }
          switch %4#0, %4#1 : !hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]> ^bb4
        ^bb3:  // pred: ^bb0
          %5:2 = dfg input extensions []  : () -> (!hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]>) {
            %8 = call @new  []() -> !hugr<opaque "qubit"["prelude"] [Linear]>
            %9 = make_tuple()
            %10 = tag 0 %9 : tuple<> -> <tuple<>>
            output %10, %8 : !hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]>
          }
          switch %5#0, %5#1 : !hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]> ^bb4
        ^bb4(%6: !hugr<opaque "qubit"["prelude"] [Linear]>):  // 2 preds: ^bb2, ^bb3
          %7:2 = dfg input extensions [] %6 : (!hugr<opaque "qubit"["prelude"] [Linear]>) -> (!hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]>) {
          ^bb0(%arg3: !hugr<opaque "qubit"["prelude"] [Linear]>):
            %8 = make_tuple()
            %9 = tag 0 %8 : tuple<> -> <tuple<>>
            output %9, %arg3 : !hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]>
          }
          switch %7#0, %7#1 : !hugr.sum<tuple<>>, !hugr<opaque "qubit"["prelude"] [Linear]> ^bb1
        }
        output %1 : !hugr<opaque "qubit"["prelude"] [Linear]>
      }
      output %0 : !hugr<opaque "qubit"["prelude"] [Linear]>
    }
    func @new[]() -> !hugr<opaque "qubit"["prelude"] [Linear]>
  }
}

// CHECK-LABEL: module {
// CHECK-NOT: cfg