// RUN: bishengir-opt -transform-interpreter -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @tile_insert_slice_0(
func.func @tile_insert_slice_0(%arg0: tensor<32x192xf32>,
                               %arg1: tensor<256x192xf32>) -> tensor<256x192xf32> {
  %inserted = tensor.insert_slice %arg0 into %arg1[0, 0] [32, 192] [1, 1] : tensor<32x192xf32> into tensor<256x192xf32>
  return %inserted : tensor<256x192xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %slice = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b, %c = transform.structured.tile_using_for %slice tile_sizes [2, 48]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tile_insert_slice_1(
func.func @tile_insert_slice_1(%arg0: tensor<32x192xf32>,
                               %arg1: tensor<256x192xf32>) -> tensor<256x192xf32> {
  %inserted = tensor.insert_slice %arg0 into %arg1[0, 0] [32, 192] [1, 1] : tensor<32x192xf32> into tensor<256x192xf32>
  %empty = tensor.empty() : tensor<256x192xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
                                ins(%inserted, %arg1 : tensor<256x192xf32>, tensor<256x192xf32>) 
                                outs(%empty : tensor<256x192xf32>) -> tensor<256x192xf32>
  return %mul : tensor<256x192xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.elemwise_binary"]} in %0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [16, 48] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["tensor.insert_slice"]} in %0 : (!transform.any_op) -> !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}