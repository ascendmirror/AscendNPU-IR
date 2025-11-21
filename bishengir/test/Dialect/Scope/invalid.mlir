// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// CHECK-LABEL: test_scope
func.func @test_scope() {
  %0 = arith.constant 1 : index
  scope.scope : () -> () {
  // expected-error@below {{`scope.return` op must not return anything for now}}
    scope.return %0 : index
  }
  return
}

// -----

// CHECK-LABEL: test_scope_return
func.func @test_scope_return() {
  // expected-error@below {{`scope.return` op expects parent op `scope.scope`}}
  scope.return
  return
}
