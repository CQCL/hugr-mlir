// RUN: hugr-mlir-opt %s | FileCheck %s

hugr.module attributes {
  test_attr = #hugr.constraint<Equatable>,
  test_type = !hugr.test
} {}

// CHECK-LABEL: hugr.module attributes {
// CHECK: test_attr = #hugr.constraint<Equatable>
// CHECK: test_type = !hugr.test
