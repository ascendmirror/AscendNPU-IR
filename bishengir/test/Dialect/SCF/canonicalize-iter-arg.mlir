// RUN: bishengir-opt -scf-canonicalize-iter-arg -allow-unregistered-dialect %s | FileCheck %s

func.func @test() -> (tensor<?xi8>, tensor<?xi8>) {
  %size = "some_op"() : () -> index
  %e = tensor.empty(%size) : tensor<?xi8>
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %ub = arith.constant 16 : index
  %cond = "some_op"() : () -> i1
  // CHECK: iter_args(
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]] = [[INIT:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]] = [[INIT]]
  %res:2 = scf.for %i = %lb to %ub  step %step iter_args(%arg0 = %e, %arg1 = %e) -> (tensor<?xi8>, tensor<?xi8>) {
    // CHECK: iter_args(
    // CHECK-SAME: = [[INIT]]
    %inner = scf.for %j = %lb to %ub step %step iter_args(%iarg = %arg0) -> tensor<?xi8> {
      "some_op"(): ()-> ()
      scf.yield %iarg : tensor<?xi8>
    }
    // CHECK: scf.if
    %inner2 = scf.if %cond -> tensor<?xi8> {
      "some_op"(): ()-> ()
      // CHECK: yield [[INIT]]
      scf.yield %e : tensor<?xi8>
    } else {
      // CHECK: yield [[INIT]]
      scf.yield %arg1 : tensor<?xi8>
    }

    // CHECK: yield [[INIT]], [[INIT]] 
    scf.yield %inner, %inner2 : tensor<?xi8>, tensor<?xi8>
  }
  // CHECK: return [[INIT]], [[INIT]]
  return %res#0, %res#1 : tensor<?xi8>, tensor<?xi8>
}
