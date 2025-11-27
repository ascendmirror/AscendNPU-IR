// RUN: bishengir-opt -outline-scope -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_scope_scope__scope__0(
// CHECK-SAME:                                        %[[VAL:.*]]: f32,
// CHECK-SAME:                                        %[[ALLOC:.*]]: memref<f32>) attributes {debug = 2 : index, tcore_type = #hivm.tcore_type<CUBE>} {
// CHECK: memref.store %[[VAL]], %[[ALLOC]][] {debug = 3 : index} : memref<f32>
// CHECK: return

// CHECK-LABEL: func.func @test_scope_scope(
// CHECK-SAME:                              %[[ARG:.*]]: memref<f32>) attributes {debug = 0 : index} {
// CHECK: %[[CST:.*]] = arith.constant {debug = 1 : index} 1.000000e-01 : f32
// CHECK: call @test_scope_scope__scope__0(%[[CST]], %[[ARG]]) : (f32, memref<f32>) -> ()
// CHECK: return {debug = 5 : index}

module {
  func.func @test_scope_scope(%arg0: memref<f32>) attributes {debug = 0 : index} {
    %cst = arith.constant {debug = 1 : index} 1.000000e-01 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] {debug = 3 : index} : memref<f32>
      scope.return {debug = 4 : index}
    } {debug = 2 : index, tcore_type = #hivm.tcore_type<CUBE>}
    return {debug = 5 : index}
  }
}
