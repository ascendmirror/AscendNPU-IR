// RUN: bishengir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

module {
  // CHECK-LABEL func.func @print_test
  // CHECK: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: tensor<32xf32>, %[[ARG2:.*]]: tensor<32xi8>)
  // CHECK: hfusion.print "PID: " {hex = true} %[[ARG0]] : i32
  // CHECK: hfusion.print "Val: " {hex = false} %[[ARG1]] : tensor<32xf32>
  // CHECK: hfusion.print "" {hex = false} %[[ARG2]] : tensor<32xi8>
  func.func @print_test(%arg0 : i32, %arg1 : tensor<32xf32>, %arg2 : tensor<32xi8>) {
    hfusion.print "PID: " {hex = true} %arg0 : i32
    hfusion.print "Val: " {hex = false} %arg1 : tensor<32xf32>
    hfusion.print "" {hex = false} %arg2 : tensor<32xi8>
    func.return
  }
}

// -----
// CHECK-LABEL: @test_group_matmul
func.func @test_group_matmul(%w1 : tensor<2x?x?xf32>, %tokens : tensor<?x?xf32>, %tpe : tensor<2xi64>, %out : tensor<?x?xf32>) {
  %res = hfusion.group_matmul
    ins(%w1, %tokens, %tpe : tensor<2x?x?xf32>, tensor<?x?xf32>, tensor<2xi64>)
    outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
    return
}

// -----
func.func @histogram_nomask(%arg0: tensor<8xi32>) -> tensor<4xi32> {
  // CHECK-LABEL: func.func @histogram_nomask
  // CHECK: hfusion.histogram
  // CHECK: return
  %res = hfusion.histogram %arg0, 4 : tensor<8xi32> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----
func.func @histogram_mask(%arg0: tensor<8xi32>, %mask: tensor<8xi1>)
    -> tensor<4xi32> {
  // CHECK-LABEL: func.func @histogram_mask
  // CHECK: hfusion.histogram
  // CHECK: return
  %res = hfusion.histogram %arg0, 4, %mask
         : tensor<8xi32>, tensor<8xi1> -> tensor<4xi32>
  return %res : tensor<4xi32>
}
