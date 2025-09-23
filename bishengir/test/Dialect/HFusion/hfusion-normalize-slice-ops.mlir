// RUN: bishengir-opt --hfusion-normalize-slice-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_2:.*]] = hfusion.interleave %[[VAL_0]], %[[VAL_1]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.insert_slice
func.func @test_interleave() -> tensor<4x2x128xf16> {
  %0 = tensor.empty() : tensor<4x2x64xf16>
  %1 = tensor.empty() : tensor<4x2x64xf16>
  %2 = tensor.empty() : tensor<4x2x128xf16>
  %3 = tensor.insert_slice %0 into %2[0, 0, 0] [4, 2, 64] [1, 1, 2] : tensor<4x2x64xf16> into tensor<4x2x128xf16>
  %4 = tensor.insert_slice %1 into %3[0, 0, 1] [4, 2, 64] [1, 1, 2] : tensor<4x2x64xf16> into tensor<4x2x128xf16>
  return %4 : tensor<4x2x128xf16>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK-DAG: %[[EXPAND_0:.*]] = tensor.expand_shape %[[VAL_0]] {{\[}}[0], [1, 2]] output_shape [4, 2, 1] : tensor<4x2xf16> into tensor<4x2x1xf16>
// CHECK-DAG: %[[EXPAND_1:.*]] = tensor.expand_shape %[[VAL_1]] {{\[}}[0], [1, 2]] output_shape [4, 2, 1] : tensor<4x2xf16> into tensor<4x2x1xf16>
// CHECK: hfusion.interleave %[[EXPAND_0]], %[[EXPAND_1]] : tensor<4x2x1xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.insert_slice
func.func @test_interleave_reduced_rank() -> tensor<4x2x2xf16> {
  %0 = tensor.empty() : tensor<4x2xf16>
  %1 = tensor.empty() : tensor<4x2xf16>
  %2 = tensor.empty() : tensor<4x2x2xf16>
  %3 = tensor.insert_slice %0 into %2[0, 0, 0] [4, 2, 1] [1, 1, 2] : tensor<4x2xf16> into tensor<4x2x2xf16>
  %4 = tensor.insert_slice %1 into %3[0, 0, 1] [4, 2, 1] [1, 1, 2] : tensor<4x2xf16> into tensor<4x2x2xf16>
  return %4 : tensor<4x2x2xf16>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty(%[[DIM:.*]]) : tensor<?x2xf16>
// CHECK: %[[VAL_1:.*]] = tensor.empty(%[[DIM]]) : tensor<?x2xf16>
// CHECK-DAG: %[[EXPAND_0:.*]] = tensor.expand_shape %[[VAL_0]] {{\[}}[0], [1, 2]] output_shape [%[[DIM_0:.*]], 2, 1] : tensor<?x2xf16> into tensor<?x2x1xf16>
// CHECK-DAG: %[[EXPAND_1:.*]] = tensor.expand_shape %[[VAL_1]] {{\[}}[0], [1, 2]] output_shape [%[[DIM_1:.*]], 2, 1] : tensor<?x2xf16> into tensor<?x2x1xf16>
// CHECK: hfusion.interleave %[[EXPAND_0]], %[[EXPAND_1]] : tensor<?x2x1xf16>, tensor<?x2x1xf16> -> tensor<?x2x2xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.insert_slice
func.func @test_interleave_reduced_rank(%arg0: tensor<?x2xf16>) -> tensor<?x2x2xf16> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xf16>
  %0 = tensor.empty(%dim) : tensor<?x2xf16>
  %1 = tensor.empty(%dim) : tensor<?x2xf16>
  %2 = tensor.empty(%dim) : tensor<?x2x2xf16>
  %dim_0 = tensor.dim %0, %c0 : tensor<?x2xf16>
  %dim_1 = tensor.dim %1, %c0 : tensor<?x2xf16>
  %3 = tensor.insert_slice %0 into %2[0, 0, 0] [%dim_0, 2, 1] [1, 1, 2] : tensor<?x2xf16> into tensor<?x2x2xf16>
  %4 = tensor.insert_slice %1 into %3[0, 0, 1] [%dim_1 , 2, 1] [1, 1, 2] : tensor<?x2xf16> into tensor<?x2x2xf16>
  return %4 : tensor<?x2x2xf16>
}

// -----

// CHECK-NOT: hfusion.interleave
func.func @test_not_interleave() -> tensor<4x32x64xf16> {
  %0 = tensor.empty() : tensor<4x32xf16>
  %1 = tensor.empty() : tensor<4x32xf16>
  %2 = tensor.empty() : tensor<4x32x64xf16>
  %3 = tensor.insert_slice %0 into %2[0, 0, 0] [4, 32, 1] [1, 1, 2] : tensor<4x32xf16> into tensor<4x32x64xf16>
  %4 = tensor.insert_slice %1 into %3[0, 0, 2] [4, 32, 1] [1, 1, 2] : tensor<4x32xf16> into tensor<4x32x64xf16>
  return %4 : tensor<4x32x64xf16>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x128xf16>
// CHECK: %[[VAL_1:.*]] = hfusion.deinterleave %[[VAL_0]] channel<0> : tensor<4x2x128xf16> -> tensor<4x2x64xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.extract_slice
func.func @test_deinterleave() -> tensor<4x2x64xf16> {
  %0 = tensor.empty() : tensor<4x2x128xf16>
  %1 = tensor.extract_slice %0[0, 0, 0] [4, 2, 64] [1, 1, 2] : tensor<4x2x128xf16> to tensor<4x2x64xf16>
  return %1 : tensor<4x2x64xf16>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x128xf16>
// CHECK: %[[VAL_1:.*]] = hfusion.deinterleave %[[VAL_0]] channel<1> : tensor<4x2x128xf16> -> tensor<4x2x64xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.extract_slice
func.func @test_deinterleave() -> tensor<4x2x64xf16> {
  %0 = tensor.empty() : tensor<4x2x128xf16>
  %1 = tensor.extract_slice %0[0, 0, 1] [4, 2, 64] [1, 1, 2] : tensor<4x2x128xf16> to tensor<4x2x64xf16>
  return %1 : tensor<4x2x64xf16>
}

// -----

// CHECK-DAG: %[[INPUT:.*]] = tensor.empty() : tensor<4x2x2xf16>
// CHECK: %[[OUTPUT:.*]] = hfusion.deinterleave %[[INPUT]] channel<1> : tensor<4x2x2xf16> -> tensor<4x2x1xf16>
// CHECK: %[[COLLPASE:.*]] = tensor.collapse_shape %[[OUTPUT]] {{\[}}[0], [1, 2]] : tensor<4x2x1xf16> into tensor<4x2xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.extract_slice
func.func @test_deinterleave_reduced_rank() -> tensor<4x2xf16> {
  %0 = tensor.empty() : tensor<4x2x2xf16>
  %1 = tensor.extract_slice %0[0, 0, 1] [4, 2, 1] [1, 1, 2] : tensor<4x2x2xf16> to tensor<4x2xf16>
  return %1 : tensor<4x2xf16>
}

// -----

// CHECK: %[[OUTPUT:.*]] = hfusion.deinterleave %arg0 channel<1> : tensor<?x?x2xf16> -> tensor<?x?x1xf16>
// CHECK: %[[COLLPASE:.*]] = tensor.collapse_shape %[[OUTPUT]] {{\[}}[0], [1, 2]] : tensor<?x?x1xf16> into tensor<?x?xf16>
// CHECK-NOT-NORMALIZE-SLICE: tensor.extract_slice
func.func @test_deinterleave_dynamic(%arg0: tensor<?x?x2xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_0 = tensor.dim %arg0, %c0 : tensor<?x?x2xf16>
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x2xf16>

  %0 = tensor.extract_slice %arg0[0, 0, 1] [%dim_0, %dim_1, 1] [1, 1, 2] : tensor<?x?x2xf16> to tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// CHECK-NOT: hfusion.deinterleave
func.func @test_not_deinterleave() -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4x32xf16>
  %1 = tensor.extract_slice %0[0, 0] [4, 1] [1, 2] : tensor<4x32xf16> to tensor<4xf16>
  return %1 : tensor<4xf16>
}

// -----

// CHECK: hfusion.deinterleave
// CHECK: tensor.collapse_shape
func.func @test_deinterleave_with_reduce_rank(%arg0: tensor<1x32x2xbf16>) -> tensor<1x32xbf16> {
  %0 = tensor.extract_slice %arg0[0, 0, 0] [1, 32, 1] [1, 1, 1] : tensor<1x32x2xbf16> to tensor<1x32xbf16>
  return %0 : tensor<1x32xbf16>
}


// -----

// CHECK-NOT: hfusion.deinterleave
// CHECK: tensor.extract_slice
func.func @test_not_deinterleave_with_dynamic_last_dim_offset(%arg0: tensor<1x32x2xbf16>, %offset: index) -> tensor<1x32x1xbf16> {
  %0 = tensor.extract_slice %arg0[0, 0, %offset] [1, 32, 1] [1, 1, 1] : tensor<1x32x2xbf16> to tensor<1x32x1xbf16>
  return %0 : tensor<1x32x1xbf16>
}

// -----

// CHECK-LABEL: @test_normalize_interleave_i1
// CHECK :  %[[res_f16:.*]] = hfusion.interleave %[[arg0_f16:.*]], %[[arg1_f16:.*]] : tensor<4x2x32xf16>, tensor<4x2x32xf16> -> tensor<4x2x64xf16>
func.func @test_normalize_interleave_i1(%arg0 : tensor<4x2x32xi1>, %arg1 : tensor<4x2x32xi1>) -> tensor<4x2x64xi1> {
  %0 = tensor.empty() : tensor<4x2x64xi1>
  %1 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi1>, tensor<4x2x32xi1> -> tensor<4x2x64xi1>
  return %1 : tensor<4x2x64xi1>
}
