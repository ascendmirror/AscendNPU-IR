// RUN: bishengir-opt -outline-scope -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @test_scope_scope_scope_0(
// CHECK-SAME:                                          %[[VAL_0:.*]]: f32,
// CHECK-SAME:                                          %[[ARG_0:.*]]: memref<f32>) attributes {debug = 2 : index, tcore_type = #hivm.tcore_type<CUBE>} {
// CHECK:           memref.store %[[VAL_0]], %[[ARG_0]][] {debug = 3 : index} : memref<f32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @test_scope_scope_scope_1(
// CHECK-SAME:                                          %[[VAL_1:.*]]: f32,
// CHECK-SAME:                                          %[[ARG_1:.*]]: memref<f32>) attributes {tcore_type = #hivm.tcore_type<VECTOR>} {
// CHECK:           memref.store %[[VAL_1]], %[[ARG_1]][] : memref<f32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @test_scope_scope(
// CHECK-SAME:                                %[[ALLOC:.*]]: memref<f32>) attributes {debug = 0 : index} {
// CHECK:           %[[CST:.*]] = arith.constant {debug = 1 : index} 1.000000e-01 : f32
// CHECK:           call @test_scope_scope_scope_0(%[[CST]], %[[ALLOC]]) : (f32, memref<f32>) -> ()
// CHECK:           call @test_scope_scope_scope_1(%[[CST]], %[[ALLOC]]) : (f32, memref<f32>) -> ()
// CHECK:           return {debug = 5 : index}
// CHECK:         }

module {
  func.func @test_scope_scope(%arg0: memref<f32>) attributes {debug = 0 : index} {
    %cst = arith.constant {debug = 1 : index} 1.000000e-01 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] {debug = 3 : index} : memref<f32>
      scope.return {debug = 4 : index}
    } {debug = 2 : index, tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    } {tcore_type = #hivm.tcore_type<VECTOR>}
    return {debug = 5 : index}
  }
}
